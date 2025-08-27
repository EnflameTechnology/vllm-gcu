import dataclasses
from vllm.engine.arg_utils import EngineArgs
from typing import Optional, Type
import torch
from tqdm import tqdm
import logging
import os
import sys
import csv
from datetime import datetime

from vllm_utils.evaluate_datasets.configs.models.reranker.reranker import BaseReranker, VLLMReranker
from vllm_utils.evaluate_datasets.configs.datasets.beir.beir import BEIRDataset
from vllm.utils import FlexibleArgumentParser

from rank_bm25 import BM25Okapi
import numpy as np

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

def save_results_to_csv(args: FlexibleArgumentParser, metrics: dict, csv_path: str = "evaluation_results.csv"):
    """Save both input arguments and evaluation metrics to CSV file"""
    # Prepare data for CSV
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Get all arguments as dictionary
    row_data = {
        'timestamp': timestamp,
        'model': args.model,
        'dataset': args.dataset,
    }

    # Add all input arguments
    args_dict = vars(args)
    for key, value in args_dict.items():
        if key not in ['engine_args', 'model', 'dataset']:  # Skip already added and complex args
            row_data[f'arg_{key}'] = str(value)

    # Add all metrics
    row_data.update({f'metric_{k}': f"{v:.4f}" for k, v in metrics.items()})

    # Check if file exists to determine if we need to write headers
    file_exists = os.path.isfile(csv_path)

    # Write to CSV
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row_data.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_data)

    logging.info(f"Results and parameters saved to {csv_path}")

class BEIREvaluator:
    """General BEIR evaluator"""

    def __init__(self,
                 dataset: BEIRDataset,
                 reranker: BaseReranker,
                 initial_retriever: str = "bm25",
                 initial_top_k: int = 1000,
                 rerank_top_k: int = 100):
        self.dataset = dataset
        self.reranker = reranker
        self.initial_retriever = initial_retriever
        self.initial_top_k = initial_top_k
        self.rerank_top_k = rerank_top_k

    def _build_bm25(self, corpus):
        """Build BM25 retriever"""
        documents = [doc["text"] for doc in corpus.values()]
        tokenized_documents = [doc.split() for doc in documents]
        bm25 = BM25Okapi(tokenized_documents)
        bm25.doc_ids = list(corpus.keys())
        return bm25

    def _initial_retrieve(self, corpus, queries):
        """Execute initial retrieval"""
        if self.initial_retriever == "bm25":
            bm25 = self._build_bm25(corpus)
            results = {}
            for query_id, query_text in tqdm(queries.items(), desc="BM25 Retrieval"):
                tokenized_query = query_text.split()
                doc_scores = bm25.get_scores(tokenized_query)
                ranked_indices = np.argsort(doc_scores)[::-1][:self.initial_top_k]
                results[query_id] = {
                    bm25.doc_ids[i]: float(doc_scores[i]) for i in ranked_indices
                }
            return results
        else:
            raise ValueError(f"Unsupported initial retriever: {self.initial_retriever}")

    def evaluate(self):
        """Execute evaluation"""
        # load dataset
        corpus, queries, qrels = self.dataset.load()

        # initial retrieval
        initial_results = self._initial_retrieve(corpus, queries)

        # reranking
        reranked_results = {}

        for query_id in tqdm(initial_results.keys(), desc="Reranking"):
            query_text = queries[query_id]
            doc_ids = list(initial_results[query_id].keys())[:self.rerank_top_k]
            doc_texts = [corpus[doc_id]["text"] for doc_id in doc_ids]

            scores = self.reranker.rerank(query_text, doc_texts)

            reranked = sorted(zip(doc_ids, scores), key=lambda x: -x[1])
            reranked_results[query_id] = {
                doc_id: float(score) for doc_id, score in reranked
            }

        # evaluate results
        initial_metrics = self.dataset.evaluate(initial_results, qrels)
        rerank_metrics = self.dataset.evaluate(reranked_results, qrels)

        return initial_metrics, rerank_metrics

def main(args: FlexibleArgumentParser):
    """Main function that takes args as parameter"""
    logging.info(f"Running with arguments: {vars(args)}")

    # create vLLM engine parameters
    engine_args = EngineArgs.from_cli_args(args)
    args.engine_args = engine_args  # Store for logging

    # initialize dataset
    dataset = BEIRDataset(
        dataset_name=args.dataset,
        data_path=args.beir_data_root,
        k_values=[5, 10]
    )

    # initialize reranker model
    reranker = VLLMReranker(engine_args)

    # create evaluator and execute evaluation
    evaluator = BEIREvaluator(
        dataset=dataset,
        reranker=reranker,
        initial_top_k=args.initial_top_k,
        rerank_top_k=args.rerank_top_k
    )

    # execute evaluation
    initial_metrics, rerank_metrics = evaluator.evaluate()

    # output results
    logging.info("\n=== Initial Retrieval Results ===")
    for k, v in initial_metrics.items():
        logging.info(f"{k}: {v:.4f}")

    logging.info("\n=== Reranking Results ===")
    for k, v in rerank_metrics.items():
        logging.info(f"{k}: {v:.4f}")

    # save results and parameters to CSV
    save_results_to_csv(
        args=args,
        metrics=rerank_metrics,
        csv_path=args.csv_path
    )

    return initial_metrics, rerank_metrics

if __name__ == '__main__':
    # create parser
    parser = FlexibleArgumentParser(description='Evaluate rerankers on BEIR datasets')

    # add evaluation related parameters (before vLLM parameters)
    parser.add_argument('--dataset', type=str, required=True,
                       choices=sorted(list(BEIRDataset.SUPPORTED_DATASETS)),
                       help='BEIR dataset name')
    parser.add_argument('--beir-data-root', type=str, required=True,
                       help='Root directory containing BEIR datasets')
    parser.add_argument('--initial-top-k', type=int, default=1000,
                       help='Number of documents to retrieve initially')
    parser.add_argument('--rerank-top-k', type=int, default=100,
                       help='Number of documents to rerank')
    parser.add_argument('--csv-path', type=str, default='evaluation_results.csv',
                       help='Path to save evaluation results CSV file')

    # add vLLM engine parameters (it will automatically add trust_remote_code)
    parser = EngineArgs.add_cli_args(parser)

    args = parser.parse_args()
    main(args)