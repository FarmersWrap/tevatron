import pickle

import numpy as np
import glob
from argparse import ArgumentParser
from collections import defaultdict
from itertools import chain
from tqdm import tqdm
import faiss

from tevatron.retriever.searcher import FaissFlatSearcher

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def search_queries(retriever, q_reps, p_lookup, args):
    if args.batch_size > 0:
        all_scores, all_indices = retriever.batch_search(q_reps, args.depth, args.batch_size, args.quiet)
    else:
        all_scores, all_indices = retriever.search(q_reps, args.depth)

    psg_indices = [[str(p_lookup[x]) for x in q_dd] for q_dd in all_indices]
    psg_indices = np.array(psg_indices)
    return all_scores, psg_indices


def load_qrels(qrels_path):
    """Load qrels file. Returns dict: {query_id: {doc_id: relevance}}."""
    qrels = defaultdict(dict)
    with open(qrels_path) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 4:
                qid, _, doc_id, rel = parts
            elif len(parts) == 3:
                qid, doc_id, rel = parts
            else:
                continue
            qrels[qid][doc_id] = int(rel)
    return qrels


def search_queries_chunked(retriever, q_reps, p_lookup, args, q_lookup=None, qrels=None):
    """
    Search with chunked passages and aggregate by document using MaxSim.
    If qrels is provided, also tracks per-chunk scores for positive passages.
    Returns (aggregated_results, chunk_selections) where chunk_selections is a list of dicts
    for positive passages only, or None if qrels is not provided.
    """
    # Search more chunks to ensure good recall after aggregation
    chunk_multiplier = getattr(args, 'chunk_multiplier', 10)
    search_depth = args.depth * chunk_multiplier

    if args.batch_size > 0:
        # all_scores.shape = [Q, search_depth]
        all_scores, all_indices = retriever.batch_search(q_reps, search_depth, args.batch_size, args.quiet)
    else:
        # all_scores.shape = [search_depth]
        all_scores, all_indices = retriever.search(q_reps, search_depth)

    # Build doc->chunk_count map for total chunks per doc
    doc_total_chunks = defaultdict(int)
    for doc_id, chunk_idx in p_lookup:
        doc_total_chunks[doc_id] = max(doc_total_chunks[doc_id], chunk_idx + 1)

    # Aggregate by document ID using MaxSim
    aggregated_results = []
    chunk_selections = [] if qrels is not None else None

    for q_idx in range(len(q_reps)):
        scores = all_scores[q_idx]
        indices = all_indices[q_idx]
        doc_max_scores = defaultdict(lambda: float('-inf'))
        doc_best_chunk = {}  # doc_id -> (best_chunk_idx, best_score)
        doc_all_chunks = defaultdict(list)  # doc_id -> [(chunk_idx, score), ...]

        for score, idx in zip(scores, indices):
            if idx < 0:  # FAISS returns -1 for insufficient results
                continue
            if idx >= len(p_lookup):  # Boundary check: prevent IndexError
                logger.warning(f"Index {idx} out of bounds for p_lookup (length {len(p_lookup)}), skipping")
                continue

            try:
                doc_id, chunk_idx = p_lookup[idx]
            except (ValueError, TypeError) as e:
                logger.error(f"p_lookup[{idx}] is not a tuple (doc_id, chunk_idx): {p_lookup[idx]}, error: {e}")
                continue

            doc_all_chunks[doc_id].append((chunk_idx, float(score)))

            # MaxSim: keep the maximum score for each document
            if score > doc_max_scores[doc_id]:
                doc_max_scores[doc_id] = score
                doc_best_chunk[doc_id] = (chunk_idx, float(score))

        # Sort by score and take top-depth
        sorted_docs = sorted(doc_max_scores.items(), key=lambda x: x[1], reverse=True)[:args.depth]
        aggregated_results.append(sorted_docs)

        # Record chunk selections for positive passages
        if qrels is not None and q_lookup is not None:
            qid = q_lookup[q_idx]
            pos_docs = {d for d, r in qrels.get(qid, {}).items() if r > 0}
            for doc_id in pos_docs:
                if doc_id in doc_best_chunk:
                    best_chunk_idx, best_score = doc_best_chunk[doc_id]
                    total_chunks = doc_total_chunks[doc_id]
                    all_chunk_scores = sorted(doc_all_chunks[doc_id], key=lambda x: x[0])
                    chunk_selections.append({
                        'qid': qid,
                        'doc_id': doc_id,
                        'best_chunk_idx': best_chunk_idx,
                        'best_score': best_score,
                        'total_chunks': total_chunks,
                        'normalized_pos': best_chunk_idx / max(total_chunks - 1, 1),
                        'all_chunk_scores': all_chunk_scores,
                    })

    return aggregated_results, chunk_selections


def write_ranking(corpus_indices, corpus_scores, q_lookup, ranking_save_file):
    with open(ranking_save_file, 'w') as f:
        for qid, q_doc_scores, q_doc_indices in zip(q_lookup, corpus_scores, corpus_indices):
            score_list = [(s, idx) for s, idx in zip(q_doc_scores, q_doc_indices)]
            score_list = sorted(score_list, key=lambda x: x[0], reverse=True)
            for s, idx in score_list:
                f.write(f'{qid}\t{idx}\t{s}\n')


def write_ranking_chunked(results, q_lookup, ranking_save_file):
    """
    Write ranking results from chunked search.
    results: List[List[Tuple[doc_id, score]]]
    """
    with open(ranking_save_file, 'w') as f:
        for qid, doc_scores in zip(q_lookup, results):
            for doc_id, score in doc_scores:
                f.write(f'{qid}\t{doc_id}\t{score}\n')


def pickle_load(path):
    with open(path, 'rb') as f:
        reps, lookup = pickle.load(f)
    return np.array(reps), lookup


def pickle_save(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def main():
    parser = ArgumentParser()
    parser.add_argument('--query_reps', required=True)
    parser.add_argument('--passage_reps', required=True)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--depth', type=int, default=1000)
    parser.add_argument('--save_ranking_to', required=True)
    parser.add_argument('--save_text', action='store_true')
    parser.add_argument('--quiet', action='store_true')
    # Chunked search arguments
    parser.add_argument('--chunked', action='store_true',
                        help='Enable chunked search with document-level MaxSim aggregation')
    parser.add_argument('--chunk_multiplier', type=int, default=10,
                        help='Multiply search depth by this factor for chunked search to ensure recall')
    parser.add_argument('--qrels', type=str, default=None,
                        help='Path to qrels file. When provided with --chunked, saves per-chunk '
                             'selection stats for positive passages to <save_ranking_to>.chunk_stats.tsv')

    args = parser.parse_args()

    index_files = glob.glob(args.passage_reps)
    logger.info(f'Pattern match found {len(index_files)} files; loading them into index.')

    p_reps_0, p_lookup_0 = pickle_load(index_files[0])
    retriever = FaissFlatSearcher(p_reps_0)

    shards = chain([(p_reps_0, p_lookup_0)], map(pickle_load, index_files[1:]))
    if len(index_files) > 1:
        shards = tqdm(shards, desc='Loading shards into index', total=len(index_files))
    look_up = []
    for p_reps, p_lookup in shards:
        retriever.add(p_reps)
        look_up += p_lookup

    # Auto-detect chunked format: lookup entries are tuples (doc_id, chunk_idx)
    is_chunked = args.chunked or (len(look_up) > 0 and isinstance(look_up[0], tuple))
    if is_chunked:
        unique_docs = len(set(doc_id for doc_id, _ in look_up))
        logger.info(f"Chunked mode: {len(look_up)} chunks from {unique_docs} documents")
        logger.info(f"Search depth: {args.depth} docs, chunk search depth: {args.depth * args.chunk_multiplier}")

    q_reps, q_lookup = pickle_load(args.query_reps)
    q_reps = q_reps

    num_gpus = faiss.get_num_gpus()
    if num_gpus == 0:
        logger.info("No GPU found or using faiss-cpu. Back to CPU.")
    else:
        logger.info(f"Using {num_gpus} GPU")
        if num_gpus == 1:
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True
            res = faiss.StandardGpuResources()
            retriever.index = faiss.index_cpu_to_gpu(res, 0, retriever.index, co)
        else:
            co = faiss.GpuMultipleClonerOptions()
            co.shard = True
            co.useFloat16 = True
            retriever.index = faiss.index_cpu_to_all_gpus(retriever.index, co,
                                                     ngpu=num_gpus)

    logger.info('Index Search Start')
    
    # Load qrels if provided
    qrels = None
    if args.qrels:
        qrels = load_qrels(args.qrels)
        logger.info(f"Loaded qrels: {len(qrels)} queries with relevance judgments")

    if is_chunked:
        # Chunked search with MaxSim aggregation
        aggregated_results, chunk_selections = search_queries_chunked(
            retriever, q_reps, look_up, args, q_lookup=q_lookup, qrels=qrels
        )
        logger.info('Index Search Finished (chunked mode with MaxSim aggregation)')

        if args.save_text:
            write_ranking_chunked(aggregated_results, q_lookup, args.save_ranking_to)
        else:
            # Convert to arrays for pickle
            all_scores = []
            all_doc_ids = []
            for doc_scores in aggregated_results:
                scores = [s for _, s in doc_scores]
                doc_ids = [d for d, _ in doc_scores]
                all_scores.append(scores)
                all_doc_ids.append(doc_ids)
            pickle_save((all_scores, all_doc_ids), args.save_ranking_to)

        # Save chunk selection stats for positive passages
        if chunk_selections:
            stats_path = args.save_ranking_to + '.chunk_stats.tsv'
            with open(stats_path, 'w') as f:
                f.write('qid\tdoc_id\tbest_chunk_idx\ttotal_chunks\tnormalized_pos\tbest_score\tall_chunk_scores\n')
                for sel in chunk_selections:
                    chunk_scores_str = ';'.join(f"{ci}:{s:.4f}" for ci, s in sel['all_chunk_scores'])
                    f.write(f"{sel['qid']}\t{sel['doc_id']}\t{sel['best_chunk_idx']}\t"
                            f"{sel['total_chunks']}\t{sel['normalized_pos']:.4f}\t"
                            f"{sel['best_score']:.4f}\t{chunk_scores_str}\n")
            logger.info(f"Saved chunk selection stats for {len(chunk_selections)} positive passages to {stats_path}")

            # Print summary stats
            positions = [s['normalized_pos'] for s in chunk_selections]
            best_indices = [s['best_chunk_idx'] for s in chunk_selections]
            total_chunks_list = [s['total_chunks'] for s in chunk_selections]
            logger.info(f"--- Chunk Selection Summary (positive passages) ---")
            logger.info(f"  Passages analyzed: {len(chunk_selections)}")
            logger.info(f"  Avg total chunks/doc: {np.mean(total_chunks_list):.1f} (std={np.std(total_chunks_list):.1f})")
            logger.info(f"  Selected chunk index: mean={np.mean(best_indices):.2f}, std={np.std(best_indices):.2f}")
            logger.info(f"  Normalized position:  mean={np.mean(positions):.4f}, std={np.std(positions):.4f}")
            # First-chunk selection rate
            first_rate = sum(1 for i in best_indices if i == 0) / len(best_indices)
            # Last-chunk selection rate
            last_rate = sum(1 for s in chunk_selections if s['best_chunk_idx'] == s['total_chunks'] - 1) / len(chunk_selections)
            logger.info(f"  First-chunk selection rate: {first_rate:.4f}")
            logger.info(f"  Last-chunk selection rate:  {last_rate:.4f}")
    else:
        # Standard search
        all_scores, psg_indices = search_queries(retriever, q_reps, look_up, args)
        logger.info('Index Search Finished')

        if args.save_text:
            write_ranking(psg_indices, all_scores, q_lookup, args.save_ranking_to)
        else:
            pickle_save((all_scores, psg_indices), args.save_ranking_to) 

if __name__ == '__main__':
    main()
