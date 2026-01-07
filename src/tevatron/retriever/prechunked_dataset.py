"""
Dataset class for pre-chunked training data.

This dataset class works with data formatted by format_train_prechunked.py,
where passages are already pre-chunked into lists of strings.
"""

import random
from typing import List, Tuple
from datasets import load_dataset
from torch.utils.data import Dataset

from tevatron.retriever.arguments import DataArguments

import logging

logger = logging.getLogger(__name__)


class PreChunkedTrainDataset(Dataset):
    """
    Dataset for training with pre-chunked passages.
    
    Expects data format:
    {
        "query": str,
        "passages": List[List[str]],  # List of passages, each passage is a list of chunks
        "query_id": str (optional),
        "num_positive": int (optional)
    }
    """

    def __init__(self,
                 data_args: DataArguments,
                 trainer=None,
                 dataset_name=None,
                 dataset_path=None):
        self.data_args = data_args
        self.trainer = trainer

        # Load training data
        self.train_data = load_dataset(
            dataset_name if dataset_name else self.data_args.dataset_name,
            self.data_args.dataset_config,
            data_files=dataset_path if dataset_path else self.data_args.dataset_path,
            split=self.data_args.dataset_split,
            cache_dir=self.data_args.dataset_cache_dir,
            num_proc=self.data_args.num_proc,
        )

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, item):
        group = self.train_data[item]
        epoch = int(self.trainer.state.epoch) if self.trainer else 0
        _hashed_seed = hash(item + (self.trainer.args.seed if self.trainer else 0))

        # Extract query
        query_text = group.get('query', '')
        query_image = query_video = query_audio = None
        formatted_query = (self.data_args.query_prefix + query_text,
                          query_image, query_video, query_audio)

        # Extract pre-chunked passages
        all_passages_chunks = group.get('passages', [])  # List[List[str]]
        num_positive = group.get('num_positive', len(all_passages_chunks))
        
        if not all_passages_chunks:
            # Fallback: return empty chunks list
            return formatted_query, []

        # Separate positive and negative passages
        positive_chunks = all_passages_chunks[:num_positive] if num_positive > 0 else []
        negative_chunks = all_passages_chunks[num_positive:] if num_positive < len(all_passages_chunks) else []

        formatted_documents = []  # List[List[str]] - each element is a list of chunks for one passage
        
        # Select positive passage (list of chunks)
        if positive_chunks:
            selected_positive_idx = (_hashed_seed + epoch) % len(positive_chunks)
            selected_positive_chunks = positive_chunks[selected_positive_idx]
            # Append the list of chunks directly (not as a tuple)
            formatted_documents.append(selected_positive_chunks)

        # Select negative passages
        negative_size = self.data_args.train_group_size - 1
        if len(negative_chunks) < negative_size:
            logger.warning(f"Not enough negative passages. Requested {negative_size}, got {len(negative_chunks)}")
            selected_negative_chunks = random.choices(negative_chunks, k=negative_size) if negative_chunks else []
        elif self.data_args.train_group_size == 1:
            selected_negative_chunks = []
        else:
            offset = epoch * negative_size % len(negative_chunks) if negative_chunks else 0
            selected_negative_chunks = list(negative_chunks) if negative_chunks else []
            random.Random(_hashed_seed).shuffle(selected_negative_chunks)
            selected_negative_chunks = selected_negative_chunks * 2
            selected_negative_chunks = selected_negative_chunks[offset: offset + negative_size]

        for negative_chunks_list in selected_negative_chunks:
            # Append the list of chunks directly (not as a tuple)
            formatted_documents.append(negative_chunks_list)

        return formatted_query, formatted_documents

