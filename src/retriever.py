"""
Retriever: Implements hybrid retrieval (dense + keyword + RRF fusion).
"""

from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
import numpy as np
from .vector_store import VectorStore


class Retriever:
    """Hybrid retriever combining dense semantic search and keyword search with RRF."""
    
    def __init__(self, vector_store: VectorStore):
        """
        Initialize the retriever.
        
        Args:
            vector_store: VectorStore instance
        """
        self.vector_store = vector_store
        self.bm25 = None
        self._build_keyword_index()
    
    def _build_keyword_index(self):
        """Build BM25 keyword index from all documents."""
        all_docs = self.vector_store.get_all_documents()
        if all_docs:
            # Tokenize documents for BM25
            tokenized_docs = []
            self.doc_ids = []
            self.doc_summaries = []
            
            for doc in all_docs:
                summary = doc.get('summary', '')
                tokens = summary.lower().split()
                tokenized_docs.append(tokens)
                self.doc_ids.append(doc.get('doc_id', ''))
                self.doc_summaries.append(summary)
            
            if tokenized_docs:
                self.bm25 = BM25Okapi(tokenized_docs)
    
    def dense_search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Perform dense vector search."""
        return self.vector_store.search(query, n_results=k)
    
    def keyword_search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Perform keyword search using BM25."""
        if not self.bm25:
            return []
        
        # Tokenize query
        query_tokens = query.lower().split()
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top k
        top_indices = np.argsort(scores)[::-1][:k]
        
        # Format results
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include documents with positive scores
                doc_id = self.doc_ids[idx]
                original_content = self.vector_store.get_original_content(doc_id)
                results.append({
                    'id': doc_id,
                    'score': float(scores[idx]),
                    'document': self.doc_summaries[idx],
                    'original_content': original_content.get('original_content') if original_content else None,
                    'page': original_content.get('page') if original_content else None,
                    'type': original_content.get('type') if original_content else None,
                    'metadata': original_content.get('metadata', {}) if original_content else {}
                })
        
        return results
    
    def reciprocal_rank_fusion(self, results_list: List[List[Dict[str, Any]]], k: int = 60) -> List[Dict[str, Any]]:
        """
        Fuse multiple result lists using Reciprocal Rank Fusion (RRF).
        
        Args:
            results_list: List of result lists from different retrieval methods
            k: RRF constant (typically 60)
        
        Returns:
            Fused and ranked results
        """
        doc_scores = {}
        doc_info = {}
        
        # Aggregate scores from all result lists
        for results in results_list:
            for rank, result in enumerate(results, start=1):
                doc_id = result['id']
                rrf_score = 1 / (k + rank)
                
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = 0
                    doc_info[doc_id] = result
                
                doc_scores[doc_id] += rrf_score
        
        # Sort by combined RRF score
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Format results
        fused_results = []
        for doc_id, score in sorted_docs:
            result = doc_info[doc_id].copy()
            result['rrf_score'] = score
            fused_results.append(result)
        
        return fused_results
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform hybrid retrieval using dense + keyword + RRF.
        
        Args:
            query: Search query
            top_k: Number of final results to return
        
        Returns:
            List of retrieved documents with original content
        """
        # Perform dense search
        dense_results = self.dense_search(query, k=top_k * 2)
        
        # Perform keyword search
        keyword_results = self.keyword_search(query, k=top_k * 2)
        
        # Fuse results using RRF
        fused_results = self.reciprocal_rank_fusion([dense_results, keyword_results])
        
        # Return top k
        return fused_results[:top_k]
    
    def rebuild_index(self):
        """Rebuild the keyword index (call after adding new documents)."""
        self._build_keyword_index()

