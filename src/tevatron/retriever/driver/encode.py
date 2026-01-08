import json
import logging
import os
import pickle
import sys
from contextlib import nullcontext

import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F

from rich import print

from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import (
    HfArgumentParser,
)

from tevatron.retriever.arguments import ModelArguments, DataArguments, \
    TevatronTrainingArguments as TrainingArguments
from tevatron.retriever.dataset import EncodeDataset
from tevatron.retriever.collator import EncodeCollator, ChunkedEncodeCollator, PreChunkedEncodeCollator
from tevatron.retriever.modeling import EncoderOutput, DenseModel

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments

    if training_args.local_rank > 0 or training_args.n_gpu > 1:
        raise NotImplementedError('Multi-GPU encoding is not supported.')

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )


    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    tokenizer.eos_token_id = tokenizer.pad_token_id
    if data_args.padding_side == 'right':
        tokenizer.padding_side = 'right'
    else:
        tokenizer.padding_side = 'left'

    if training_args.bf16:
        torch_dtype = torch.bfloat16
    elif training_args.fp16:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32
    
    model = DenseModel.load(
        model_args.model_name_or_path,
        pooling=model_args.pooling,
        normalize=model_args.normalize,
        lora_name_or_path=model_args.lora_name_or_path,
        cache_dir=model_args.cache_dir,
        torch_dtype=torch_dtype,
        attn_implementation=model_args.attn_implementation,
    )

    encode_dataset = EncodeDataset(
        data_args=data_args,
    )

    # Enable chunked mode if passage_chunk_size > 0 OR passage_chunk_size_range is set
    use_chunked = not data_args.encode_is_query and (data_args.passage_chunk_size > 0 or data_args.passage_chunk_size_range is not None)
    use_pre_chunked = not data_args.encode_is_query and data_args.encode_use_pre_chunked
    print("data_args.encode_is_query: ", data_args.encode_is_query)
    print("data_args.passage_chunk_size: ", data_args.passage_chunk_size)
    print("data_args.encode_use_pre_chunked: ", data_args.encode_use_pre_chunked)
    print("use_chunked: ", use_chunked)
    print("use_pre_chunked: ", use_pre_chunked)
    
    # Determine chunking mode for stats tracking
    chunking_mode = "none"
    if use_pre_chunked:
        logger.info("Using pre-chunked passage encoding (custom EOS positions from pre-chunked data)")
        model.passage_chunk_size = 1  # Signal to use chunked encoding
        encode_collator = PreChunkedEncodeCollator(data_args=data_args, tokenizer=tokenizer)
        chunking_mode = "pre_chunked"
    elif use_chunked:
        # Check for random chunking modes
        if data_args.passage_chunk_size_range is not None:
            if data_args.passage_chunk_size_variable:
                logger.info(f"Using fully random chunked encoding with range={data_args.passage_chunk_size_range}")
                chunking_mode = f"fully_random_{data_args.passage_chunk_size_range}"
            else:
                logger.info(f"Using passage-level random chunked encoding with range={data_args.passage_chunk_size_range}")
                chunking_mode = f"passage_random_{data_args.passage_chunk_size_range}"
        else:
            logger.info(f"Using fixed chunked passage encoding with chunk_size={data_args.passage_chunk_size}")
            chunking_mode = f"fixed_chunk_size_{data_args.passage_chunk_size}"
        model.passage_chunk_size = data_args.passage_chunk_size if data_args.passage_chunk_size > 0 else 1
        encode_collator = ChunkedEncodeCollator(data_args=data_args, tokenizer=tokenizer)
    else:
        encode_collator = EncodeCollator(data_args=data_args, tokenizer=tokenizer)
        chunking_mode = "no_chunking"

    encode_loader = DataLoader(
        encode_dataset,
        batch_size=training_args.per_device_eval_batch_size,
        collate_fn=encode_collator,
        shuffle=False,
        drop_last=False,
        num_workers=training_args.dataloader_num_workers,
    )
    encoded = []
    lookup_indices = []
    model = model.to(training_args.device)
    model.eval()

    # Track EOS statistics
    eos_stats = {
        'chunking_mode': chunking_mode,
        'per_passage': [],  # List of (doc_id, num_eos) tuples
        'total_eos': 0,
        'total_passages': 0,
        'config': {
            'passage_chunk_size': data_args.passage_chunk_size if hasattr(data_args, 'passage_chunk_size') else None,
            'passage_max_len': data_args.passage_max_len if hasattr(data_args, 'passage_max_len') else None,
            'encode_use_pre_chunked': data_args.encode_use_pre_chunked if hasattr(data_args, 'encode_use_pre_chunked') else False,
        }
    }

    for batch in tqdm(encode_loader):
        with torch.amp.autocast('cuda') if training_args.fp16 or training_args.bf16 else nullcontext():
            with torch.no_grad():
                if use_pre_chunked or use_chunked:
                    doc_ids, batch_inputs, eos_positions = batch
                    # batch_inputs: input_ids, attention_mask
                    for k, v in batch_inputs.items():
                        batch_inputs[k] = v.to(training_args.device)
                    
                    # Count EOS tokens per passage
                    for i, doc_id in enumerate(doc_ids):
                        num_eos = len(eos_positions[i]) if i < len(eos_positions) else 0
                        eos_stats['per_passage'].append((doc_id, num_eos))
                        eos_stats['total_eos'] += num_eos
                        eos_stats['total_passages'] += 1
                    
                    chunk_embs, chunk_mask = model.encode_passage(batch_inputs, eos_positions)
                    # chunk_embs: [batch_size, max_chunks, hidden_size]
                    # chunk_mask: [batch_size, max_chunks]
                    batch_size, max_chunks, hidden_size = chunk_embs.shape
                    for i, doc_id in enumerate(doc_ids):
                        for chunk_idx in range(max_chunks):
                            if chunk_mask[i, chunk_idx] > 0:  # Valid chunk
                                encoded.append(chunk_embs[i, chunk_idx].cpu().detach().numpy())
                                lookup_indices.append((doc_id, chunk_idx))
                else:
                    batch_ids, batch_inputs = batch
                    lookup_indices.extend(batch_ids)
                    
                    for k, v in batch_inputs.items():
                        batch_inputs[k] = v.to(training_args.device)
                    
                    if data_args.encode_is_query:
                        model_output: EncoderOutput = model(query=batch_inputs)
                        encoded.append(model_output.q_reps.cpu().detach().numpy())
                    else:
                        model_output: EncoderOutput = model(passage=batch_inputs)
                        encoded.append(model_output.p_reps.cpu().detach().numpy())
    if use_pre_chunked or use_chunked:
        # Combine encoded embeddings
        encoded = np.stack(encoded)
        logger.info(f"Encoded {len(set(d for d, c in lookup_indices))} docs into {len(lookup_indices)} chunks")
        
        # Log EOS statistics
        if eos_stats['total_passages'] > 0:
            eos_counts = [num_eos for _, num_eos in eos_stats['per_passage']]
            avg_eos = eos_stats['total_eos'] / eos_stats['total_passages']
            min_eos = min(eos_counts) if eos_counts else 0
            max_eos = max(eos_counts) if eos_counts else 0
            
            logger.info("=" * 80)
            logger.info("EOS Token Statistics:")
            logger.info(f"  Chunking Mode: {eos_stats['chunking_mode']}")
            logger.info(f"  Total passages processed: {eos_stats['total_passages']}")
            logger.info(f"  Total EOS tokens added: {eos_stats['total_eos']}")
            logger.info(f"  Average EOS per passage: {avg_eos:.2f}")
            logger.info(f"  Min EOS per passage: {min_eos}")
            logger.info(f"  Max EOS per passage: {max_eos}")
            logger.info(f"  Total chunks created: {len(lookup_indices)}")
            logger.info("=" * 80)
            
            # Save detailed EOS stats to file
            eos_stats_file = data_args.encode_output_path.replace('.pkl', '_eos_stats.json')
            eos_stats_dict = {
                'chunking_mode': eos_stats['chunking_mode'],
                'total_passages': eos_stats['total_passages'],
                'total_eos': eos_stats['total_eos'],
                'total_chunks': len(lookup_indices),
                'average_eos_per_passage': avg_eos,
                'min_eos': min_eos,
                'max_eos': max_eos,
                'config': eos_stats['config'],
                'per_passage': [{'doc_id': doc_id, 'num_eos': num_eos} 
                               for doc_id, num_eos in eos_stats['per_passage']]
            }
            with open(eos_stats_file, 'w') as f:
                json.dump(eos_stats_dict, f, indent=2)
            logger.info(f"Detailed EOS statistics saved to: {eos_stats_file}")
    else:
        encoded = np.concatenate(encoded)

    with open(data_args.encode_output_path, 'wb') as f:
        pickle.dump((encoded, lookup_indices), f)
    
    logger.info(f"Saved embeddings to {data_args.encode_output_path}, shape: {encoded.shape}")


if __name__ == "__main__":
    main()
