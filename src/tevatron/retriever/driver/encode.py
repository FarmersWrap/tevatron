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

    use_chunked = not data_args.encode_is_query and data_args.passage_chunk_size > 0
    use_pre_chunked = not data_args.encode_is_query and data_args.encode_use_pre_chunked
    use_random_chunking = not data_args.encode_is_query and data_args.passage_chunk_size_range is not None
    print("data_args.encode_is_query: ", data_args.encode_is_query)
    print("data_args.passage_chunk_size: ", data_args.passage_chunk_size)
    print("data_args.passage_chunk_size_range: ", data_args.passage_chunk_size_range)
    print("data_args.passage_chunk_size_variable: ", data_args.passage_chunk_size_variable)
    print("data_args.encode_use_pre_chunked: ", data_args.encode_use_pre_chunked)
    print("use_chunked: ", use_chunked)
    print("use_pre_chunked: ", use_pre_chunked)
    print("use_random_chunking: ", use_random_chunking)
    
    if use_pre_chunked:
        logger.info("Using pre-chunked passage encoding (custom EOS positions from pre-chunked data)")
        model.passage_chunk_size = 1  # Signal to use chunked encoding
        encode_collator = PreChunkedEncodeCollator(data_args=data_args, tokenizer=tokenizer)
    elif use_chunked or use_random_chunking:
        if use_random_chunking:
            logger.info(f"Using random chunked passage encoding with chunk_size_range={data_args.passage_chunk_size_range}, variable={data_args.passage_chunk_size_variable}")
        else:
            logger.info(f"Using chunked passage encoding with chunk_size={data_args.passage_chunk_size}")
        # For random chunking, we still need a base chunk_size for the model
        # Use the minimum of the range if random chunking is enabled
        if use_random_chunking:
            try:
                parts = [p.strip() for p in data_args.passage_chunk_size_range.split(',')]
                chunk_size_min = int(parts[0])
                model.passage_chunk_size = chunk_size_min
            except:
                model.passage_chunk_size = data_args.passage_chunk_size if data_args.passage_chunk_size > 0 else 64
        else:
            model.passage_chunk_size = data_args.passage_chunk_size
        encode_collator = ChunkedEncodeCollator(data_args=data_args, tokenizer=tokenizer)
    else:
        encode_collator = EncodeCollator(data_args=data_args, tokenizer=tokenizer)

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
    
    # Track statistics for chunked encoding
    total_chunks = 0
    doc_chunk_counts = {}  # doc_id -> number of chunks

    for batch in tqdm(encode_loader):
        with torch.amp.autocast('cuda') if training_args.fp16 or training_args.bf16 else nullcontext():
            with torch.no_grad():
                if use_pre_chunked or use_chunked or use_random_chunking:
                    doc_ids, batch_inputs, eos_positions = batch
                    # batch_inputs: input_ids, attention_mask
                    for k, v in batch_inputs.items():
                        batch_inputs[k] = v.to(training_args.device)
                    
                    chunk_embs, chunk_mask = model.encode_passage(batch_inputs, eos_positions)
                    # chunk_embs: [batch_size, max_chunks, hidden_size]
                    # chunk_mask: [batch_size, max_chunks]
                    batch_size, max_chunks, hidden_size = chunk_embs.shape
                    
                    # Log EOS count (number of chunks) for each doc in this batch
                    for i, doc_id in enumerate(doc_ids):
                        num_chunks = len(eos_positions[i]) if i < len(eos_positions) else 0
                        doc_chunk_counts[doc_id] = num_chunks
                        total_chunks += num_chunks
                        logger.debug(f"Doc {doc_id}: {num_chunks} chunks (EOS positions: {eos_positions[i] if i < len(eos_positions) else []})")
                    
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
    if use_pre_chunked or use_chunked or use_random_chunking:
        # Combine encoded embeddings
        encoded = np.stack(encoded)
        num_docs = len(set(d for d, c in lookup_indices))
        num_chunks = len(lookup_indices)
        
        # Verify total_chunks matches actual encoded chunks
        if total_chunks != num_chunks:
            logger.warning(f"Total chunks count mismatch: counted {total_chunks} EOS tokens but encoded {num_chunks} chunks")
        
        # Log summary statistics
        logger.info(f"Encoded {num_docs} docs into {num_chunks} chunks")
        logger.info(f"Total chunks in corpus dataset: {num_chunks}")
        if num_docs > 0:
            logger.info(f"Average chunks per doc: {num_chunks / num_docs:.2f}")
        
        # Log chunk distribution statistics
        if doc_chunk_counts:
            chunk_counts_list = list(doc_chunk_counts.values())
            min_chunks = min(chunk_counts_list)
            max_chunks = max(chunk_counts_list)
            mean_chunks = sum(chunk_counts_list) / len(chunk_counts_list)
            logger.info(f"Chunks per doc statistics - Min: {min_chunks}, Max: {max_chunks}, Mean: {mean_chunks:.2f}")
        
        # Log first few docs as examples
        if doc_chunk_counts:
            logger.info("Sample doc chunk counts (first 10):")
            for i, (doc_id, count) in enumerate(list(doc_chunk_counts.items())[:10]):
                logger.info(f"  Doc {doc_id}: {count} chunks (EOS tokens)")
        
        logger.info(f"Encoded embeddings shape: {encoded.shape}")
    else:
        encoded = np.concatenate(encoded)

    with open(data_args.encode_output_path, 'wb') as f:
        pickle.dump((encoded, lookup_indices), f)
    
    logger.info(f"Saved embeddings to {data_args.encode_output_path}, shape: {encoded.shape}")


if __name__ == "__main__":
    main()
