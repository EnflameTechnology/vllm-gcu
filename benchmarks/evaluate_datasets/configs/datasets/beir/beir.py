from typing import Dict, Optional, List
from beir import util
from beir.datasets.data_loader import GenericDataLoader
import os
import logging
from abc import ABC, abstractmethod

class BaseDataset(ABC):
    """Basic dataset class"""
    @abstractmethod
    def load(self):
        """Load dataset"""
        pass

    @abstractmethod
    def evaluate(self, predictions: Dict, references: Dict):
        """Evaluate results"""
        pass

class BEIRDataset(BaseDataset):
    """BEIR Dataset implementation"""

    SUPPORTED_DATASETS = {
        'msmarco', 'trec-covid', 'nfcorpus', 'bioasq', 'nq',
        'hotpotqa', 'fiqa', 'signal1m', 'trec-news', 'arguana',
        'webis-touche2020', 'cqadupstack', 'quora', 'dbpedia-entity',
        'scidocs', 'fever', 'climate-fever', 'scifact'
    }

    def __init__(self,
                 dataset_name: str,
                 data_path: str,
                 split: str = "test",
                 k_values: List[int] = [1, 3, 5, 10, 20, 100]):
        if dataset_name not in self.SUPPORTED_DATASETS:
            raise ValueError(f"Dataset {dataset_name} not supported. "
                           f"Available datasets: {sorted(list(self.SUPPORTED_DATASETS))}")

        self.dataset_name = dataset_name
        self.data_path = os.path.join(data_path, dataset_name)
        self.split = split
        self.k_values = k_values

        if not os.path.exists(self.data_path):
            raise ValueError(f"Dataset path not found: {self.data_path}")

    def load(self):
        """Load dataset"""
        try:
            loader = GenericDataLoader(data_folder=self.data_path)
            corpus, queries, qrels = loader.load(split=self.split)

            logging.info(f"Loaded {self.dataset_name} dataset:")
            logging.info(f"- Number of documents: {len(corpus)}")
            logging.info(f"- Number of queries: {len(queries)}")
            logging.info(f"- Number of relevance judgments: {sum(len(qrel) for qrel in qrels.values())}")

            return corpus, queries, qrels
        except Exception as e:
            raise Exception(f"Failed to load dataset {self.dataset_name} from {self.data_path}: {str(e)}")

    def evaluate(self, predictions: Dict, references: Dict):
        """Evaluate results"""
        from beir.retrieval.evaluation import EvaluateRetrieval
        evaluator = EvaluateRetrieval()

        ndcg, _map, recall, precision = evaluator.evaluate(
            references, predictions, self.k_values)

        metrics = {}
        for metric_name, metric_values in zip(
            ["NDCG", "MAP", "Recall", "P"],
            [ndcg, _map, recall, precision]
        ):
            for k in self.k_values:
                key = f"{metric_name}@{k}"
                metrics[key] = metric_values.get(key, 0.0)
        return metrics
