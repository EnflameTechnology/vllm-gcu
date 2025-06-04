from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Tuple
from vllm import LLM
from vllm.engine.arg_utils import EngineArgs
import torch
import dataclasses
from transformers import AutoTokenizer

class BaseReranker(ABC):
    """Base class for reranker"""

    @abstractmethod
    def rerank(self,
               query: str,
               documents: List[str],
               top_k: Optional[int] = None) -> List[float]:
        """Reranking method"""
        pass

    @abstractmethod
    def batch_rerank(self,
                     queries: List[str],
                     documents_list: List[List[str]]) -> List[List[float]]:
        """Batch reranking method"""
        pass

    @abstractmethod
    def get_score_range(self) -> Tuple[float, float]:
        """Get the score range of the model"""
        pass

class VLLMReranker(BaseReranker):
    def __init__(self, engine_args: EngineArgs):
        self.llm = LLM(
            model=engine_args.model,
            trust_remote_code=True,
            task="score"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(engine_args.model)
        self.max_seq_len = engine_args.max_model_len

    def get_score_range(self) -> Tuple[float, float]:
        return float('-inf'), float('inf')

    def rerank(self,
               query: str,
               documents: List[str],
               top_k: Optional[int] = None,
               batch_size: int = 32) -> List[float]:
        """Rerank documents and return a list of scores"""
        # Text truncation logic
        query_tokens = self.tokenizer.tokenize(query)
        query_truncated = self.tokenizer.convert_tokens_to_string(
            query_tokens[: self.max_seq_len // 2]
        )

        # Calculate remaining available length
        query_len = len(self.tokenizer.tokenize(query_truncated))
        available_len = self.max_seq_len - query_len
        if available_len < 1:
            available_len = 1

        # Truncate documents
        doc_texts_truncated = []
        for doc in documents:
            doc_tokens = self.tokenizer.tokenize(doc)
            doc_truncated = self.tokenizer.convert_tokens_to_string(
                doc_tokens[:available_len]
            )
            doc_texts_truncated.append(doc_truncated)

        # Batch scoring
        scores = []
        for i in range(0, len(doc_texts_truncated), batch_size):
            batch_docs = doc_texts_truncated[i:i+batch_size]
            res = self.llm.score(query_truncated, batch_docs, use_tqdm=False)
            batch_scores = [float(output.outputs.score) for output in res]
            scores.extend(batch_scores)

        return scores

    def batch_rerank(self,
                     queries: List[str],
                     documents_list: List[List[str]]) -> List[List[float]]:
        """Batch reranking method"""
        all_scores = []
        for query, docs in zip(queries, documents_list):
            scores = self.rerank(query, docs)
            all_scores.append(scores)
        return all_scores