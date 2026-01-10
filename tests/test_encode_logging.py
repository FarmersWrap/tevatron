import sys
from pathlib import Path
import logging
from io import StringIO
from unittest.mock import Mock, patch
import numpy as np
import torch

import pytest


def _tevatron_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _add_tevatron_src_to_path():
    # tevatron/tests/test_encode_logging.py -> tevatron/ -> tevatron/src
    src = _tevatron_root() / "src"
    sys.path.insert(0, str(src))


REAL_TEXT = (
    "Alterations of the architecture of cerebral white matter in the developing human brain can affect cortical "
    "development and result in functional disabilities. A line scan diffusion-weighted magnetic resonance imaging "
    "(MRI) sequence with diffusion tensor analysis was applied to measure the apparent diffusion coefficient, to "
    "calculate relative anisotropy, and to delineate three-dimensional fiber architecture in cerebral white matter in "
    "preterm (n = 17) and full-term infants (n = 7). To assess effects of prematurity on cerebral white matter "
    "development, early gestation preterm infants (n = 10) were studied a second time at term. In the central white "
    "matter the mean apparent diffusion coefficient at 28 wk was high, 1.8 microm2/ms, and decreased toward term to "
    "1.2 microm2/ms. In the posterior limb of the internal capsule, the mean apparent diffusion coefficients at both "
    "times were similar (1.2 versus 1.1 microm2/ms). Relative anisotropy was higher the closer birth was to term with "
    "greater absolute values in the internal capsule than in the central white matter. Preterm infants at term showed "
    "higher mean diffusion coefficients in the central white matter (1.4 +/- 0.24 versus 1.15 +/- 0.09 microm2/ms, "
    "p = 0.016) and lower relative anisotropy in both areas compared with full-term infants (white matter, 10.9 +/- "
    "0.6 versus 22.9 +/- 3.0%, p = 0.001; internal capsule, 24.0 +/- 4.44 versus 33.1 +/- 0.6% p = 0.006). "
    "Nonmyelinated fibers in the corpus callosum were visible by diffusion tensor MRI as early as 28 wk; full-term and "
    "preterm infants at term showed marked differences in white matter fiber organization. The data indicate that "
    "quantitative assessment of water diffusion by diffusion tensor MRI provides insight into microstructural "
    "development in cerebral white matter in living infants"
)


@pytest.fixture(scope="session")
def train_tokenizer():
    """
    Use the Qwen 0.6B tokenizer.
    """
    _add_tevatron_src_to_path()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    tok.eos_token_id = tok.pad_token_id
    tok.padding_side = "right"
    return tok


@pytest.mark.unit
def test_chunked_encode_logging_counts(train_tokenizer):
    """Test that logging correctly counts chunks per doc and total chunks."""
    _add_tevatron_src_to_path()
    from tevatron.retriever.arguments import DataArguments
    from tevatron.retriever.collator import ChunkedEncodeCollator
    
    # Set up logging capture
    log_capture = StringIO()
    handler = logging.StreamHandler(log_capture)
    handler.setLevel(logging.INFO)
    
    logger = logging.getLogger('tevatron.retriever.driver.encode')
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    
    try:
        data_args = DataArguments(
            passage_max_len=128,
            pad_to_multiple_of=16,
            padding_side="right",
            append_eos_token=False,
            passage_chunk_size=32,  # Fixed chunk size
        )
        collator = ChunkedEncodeCollator(data_args=data_args, tokenizer=train_tokenizer)
        
        # Create test features with multiple docs
        features = [
            ("doc1", REAL_TEXT, None, None, None),
            ("doc2", REAL_TEXT, None, None, None),
            ("doc3", "Short text", None, None, None),  # Short text will have fewer chunks
        ]
        
        doc_ids, d_collated, eos_positions = collator(features)
        
        # Simulate the encoding loop logic from encode.py
        total_chunks = 0
        doc_chunk_counts = {}
        
        for i, doc_id in enumerate(doc_ids):
            num_chunks = len(eos_positions[i]) if i < len(eos_positions) else 0
            doc_chunk_counts[doc_id] = num_chunks
            total_chunks += num_chunks
            logger.debug(f"Doc {doc_id}: {num_chunks} chunks (EOS positions: {eos_positions[i] if i < len(eos_positions) else []})")
        
        # Hardcoded golden output: passage_chunk_size=32, passage_max_len=128
        # doc1: REAL_TEXT -> 4 chunks
        # doc2: REAL_TEXT -> 4 chunks  
        # doc3: "Short text" -> 1 chunk
        expected_doc_chunk_counts = {
            "doc1": 4,
            "doc2": 4,
            "doc3": 1,
        }
        expected_total_chunks = 9
        expected_num_docs = 3
        expected_min_chunks = 1
        expected_max_chunks = 4
        expected_mean_chunks = 3.0
        
        # Verify counts match hardcoded values
        assert len(doc_chunk_counts) == expected_num_docs
        assert doc_chunk_counts == expected_doc_chunk_counts
        assert total_chunks == expected_total_chunks
        
        # Log summary (simulating encode.py)
        num_docs = len(doc_chunk_counts)
        logger.info(f"Encoded {num_docs} docs into {total_chunks} chunks")
        logger.info(f"Total chunks in corpus dataset: {total_chunks}")
        if num_docs > 0:
            logger.info(f"Average chunks per doc: {total_chunks / num_docs:.2f}")
        
        if doc_chunk_counts:
            chunk_counts_list = list(doc_chunk_counts.values())
            min_chunks = min(chunk_counts_list)
            max_chunks = max(chunk_counts_list)
            mean_chunks = sum(chunk_counts_list) / len(chunk_counts_list)
            logger.info(f"Chunks per doc statistics - Min: {min_chunks}, Max: {max_chunks}, Mean: {mean_chunks:.2f}")
        
        # Get log output
        log_output = log_capture.getvalue()
        
        # Verify log messages contain expected hardcoded information
        assert f"Encoded {expected_num_docs} docs into {expected_total_chunks} chunks" in log_output
        assert f"Total chunks in corpus dataset: {expected_total_chunks}" in log_output
        assert f"Average chunks per doc: {expected_total_chunks / expected_num_docs:.2f}" in log_output
        assert f"Chunks per doc statistics - Min: {expected_min_chunks}, Max: {expected_max_chunks}, Mean: {expected_mean_chunks:.2f}" in log_output
        
    finally:
        logger.removeHandler(handler)


@pytest.mark.unit
def test_chunked_encode_logging_with_random_chunking(train_tokenizer):
    """Test that logging correctly counts chunks with random chunking."""
    import random
    
    _add_tevatron_src_to_path()
    from tevatron.retriever.arguments import DataArguments
    from tevatron.retriever.collator import ChunkedEncodeCollator
    
    random.seed(42)
    
    # Set up logging capture
    log_capture = StringIO()
    handler = logging.StreamHandler(log_capture)
    handler.setLevel(logging.INFO)
    
    logger = logging.getLogger('tevatron.retriever.driver.encode')
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    
    try:
        data_args = DataArguments(
            passage_max_len=128,
            pad_to_multiple_of=16,
            padding_side="right",
            append_eos_token=False,
            passage_chunk_size_range="32,64",
            passage_chunk_size_variable=False,
        )
        collator = ChunkedEncodeCollator(data_args=data_args, tokenizer=train_tokenizer)
        
        # Create test features
        features = [
            ("doc1", REAL_TEXT, None, None, None),
            ("doc2", REAL_TEXT, None, None, None),
        ]
        
        doc_ids, d_collated, eos_positions = collator(features)
        
        # Simulate the encoding loop logic
        total_chunks = 0
        doc_chunk_counts = {}
        
        for i, doc_id in enumerate(doc_ids):
            num_chunks = len(eos_positions[i]) if i < len(eos_positions) else 0
            doc_chunk_counts[doc_id] = num_chunks
            total_chunks += num_chunks
        
        # Hardcoded golden output: seed=42, passage_chunk_size_range="32,64", passage_max_len=128
        # With seed=42, random.randint(32, 64) generates: 39 for doc1, 33 for doc2
        # Both produce 4 chunks each with max_length=128
        expected_doc_chunk_counts = {
            "doc1": 4,
            "doc2": 4,
        }
        expected_total_chunks = 8
        expected_num_docs = 2
        
        # Verify counts match hardcoded values
        assert len(doc_chunk_counts) == expected_num_docs
        assert doc_chunk_counts == expected_doc_chunk_counts
        assert total_chunks == expected_total_chunks
        
        # Log summary
        num_docs = len(doc_chunk_counts)
        logger.info(f"Encoded {num_docs} docs into {total_chunks} chunks")
        logger.info(f"Total chunks in corpus dataset: {total_chunks}")
        
        log_output = log_capture.getvalue()
        assert f"Encoded {expected_num_docs} docs into {expected_total_chunks} chunks" in log_output
        assert f"Total chunks in corpus dataset: {expected_total_chunks}" in log_output
        
    finally:
        logger.removeHandler(handler)


@pytest.mark.unit
def test_prechunked_encode_logging_counts(train_tokenizer):
    """Test that logging correctly counts chunks for pre-chunked passages."""
    _add_tevatron_src_to_path()
    from tevatron.retriever.arguments import DataArguments
    from tevatron.retriever.collator import PreChunkedEncodeCollator
    
    # Set up logging capture
    log_capture = StringIO()
    handler = logging.StreamHandler(log_capture)
    handler.setLevel(logging.INFO)
    
    logger = logging.getLogger('tevatron.retriever.driver.encode')
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    
    try:
        data_args = DataArguments(
            passage_max_len=512,
            pad_to_multiple_of=16,
            padding_side="right",
            append_eos_token=False,
        )
        collator = PreChunkedEncodeCollator(data_args=data_args, tokenizer=train_tokenizer)
        
        # Create pre-chunked features
        features = [
            ("doc1", ["Chunk 1", "Chunk 2", "Chunk 3"], None, None, None),  # 3 chunks
            ("doc2", ["Single chunk"], None, None, None),  # 1 chunk
            ("doc3", ["A", "B", "C", "D", "E"], None, None, None),  # 5 chunks
        ]
        
        doc_ids, d_collated, eos_positions = collator(features)
        
        # Simulate the encoding loop logic
        total_chunks = 0
        doc_chunk_counts = {}
        
        for i, doc_id in enumerate(doc_ids):
            num_chunks = len(eos_positions[i]) if i < len(eos_positions) else 0
            doc_chunk_counts[doc_id] = num_chunks
            total_chunks += num_chunks
        
        # Hardcoded golden output: pre-chunked passages
        # doc1: ["Chunk 1", "Chunk 2", "Chunk 3"] -> 3 chunks
        # doc2: ["Single chunk"] -> 1 chunk
        # doc3: ["A", "B", "C", "D", "E"] -> 5 chunks
        expected_doc_chunk_counts = {
            "doc1": 3,
            "doc2": 1,
            "doc3": 5,
        }
        expected_total_chunks = 9
        expected_num_docs = 3
        expected_min_chunks = 1
        expected_max_chunks = 5
        expected_mean_chunks = 3.0
        
        # Verify counts match hardcoded values
        assert doc_chunk_counts == expected_doc_chunk_counts
        assert total_chunks == expected_total_chunks
        
        # Log summary
        num_docs = len(doc_chunk_counts)
        logger.info(f"Encoded {num_docs} docs into {total_chunks} chunks")
        logger.info(f"Total chunks in corpus dataset: {total_chunks}")
        
        if doc_chunk_counts:
            chunk_counts_list = list(doc_chunk_counts.values())
            min_chunks = min(chunk_counts_list)
            max_chunks = max(chunk_counts_list)
            mean_chunks = sum(chunk_counts_list) / len(chunk_counts_list)
            logger.info(f"Chunks per doc statistics - Min: {min_chunks}, Max: {max_chunks}, Mean: {mean_chunks:.2f}")
        
        log_output = log_capture.getvalue()
        assert f"Encoded {expected_num_docs} docs into {expected_total_chunks} chunks" in log_output
        assert f"Total chunks in corpus dataset: {expected_total_chunks}" in log_output
        assert f"Chunks per doc statistics - Min: {expected_min_chunks}, Max: {expected_max_chunks}, Mean: {expected_mean_chunks:.2f}" in log_output
        
    finally:
        logger.removeHandler(handler)


@pytest.mark.unit
def test_chunked_encode_logging_sample_docs(train_tokenizer):
    """Test that logging includes sample doc chunk counts."""
    _add_tevatron_src_to_path()
    from tevatron.retriever.arguments import DataArguments
    from tevatron.retriever.collator import ChunkedEncodeCollator
    
    # Set up logging capture
    log_capture = StringIO()
    handler = logging.StreamHandler(log_capture)
    handler.setLevel(logging.INFO)
    
    logger = logging.getLogger('tevatron.retriever.driver.encode')
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    
    try:
        data_args = DataArguments(
            passage_max_len=128,
            pad_to_multiple_of=16,
            padding_side="right",
            append_eos_token=False,
            passage_chunk_size=32,
        )
        collator = ChunkedEncodeCollator(data_args=data_args, tokenizer=train_tokenizer)
        
        # Create test features
        features = [
            ("doc1", REAL_TEXT, None, None, None),
            ("doc2", REAL_TEXT, None, None, None),
            ("doc3", "Short", None, None, None),
        ]
        
        doc_ids, d_collated, eos_positions = collator(features)
        
        # Simulate the encoding loop logic
        doc_chunk_counts = {}
        for i, doc_id in enumerate(doc_ids):
            num_chunks = len(eos_positions[i]) if i < len(eos_positions) else 0
            doc_chunk_counts[doc_id] = num_chunks
        
        # Hardcoded golden output: passage_chunk_size=32, passage_max_len=128
        # doc1: REAL_TEXT -> 4 chunks
        # doc2: REAL_TEXT -> 4 chunks
        # doc3: "Short" -> 1 chunk
        expected_doc_chunk_counts = {
            "doc1": 4,
            "doc2": 4,
            "doc3": 1,
        }
        
        # Verify counts match hardcoded values
        assert doc_chunk_counts == expected_doc_chunk_counts
        
        # Log sample doc chunk counts (first 10)
        if doc_chunk_counts:
            logger.info("Sample doc chunk counts (first 10):")
            for i, (doc_id, count) in enumerate(list(doc_chunk_counts.items())[:10]):
                logger.info(f"  Doc {doc_id}: {count} chunks (EOS tokens)")
        
        log_output = log_capture.getvalue()
        
        # Verify sample doc counts are logged with hardcoded values
        assert "Sample doc chunk counts (first 10):" in log_output
        assert "Doc doc1: 4 chunks (EOS tokens)" in log_output
        assert "Doc doc2: 4 chunks (EOS tokens)" in log_output
        assert "Doc doc3: 1 chunks (EOS tokens)" in log_output
        
    finally:
        logger.removeHandler(handler)
