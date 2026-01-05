"""
VectorStore: Manages ChromaDB vector store for embeddings and metadata.
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import uuid


class VectorStore:
    """Manages vector storage using ChromaDB with sentence transformers."""
    
    def __init__(self, collection_name: str = "document_collection"):
        """
        Initialize the vector store.
        
        Args:
            collection_name: Name for the ChromaDB collection
        """
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_dim = 384  # Dimension for all-MiniLM-L6-v2
        
        # Initialize ChromaDB client (in-memory, ephemeral)
        self.client = chromadb.Client(Settings(
            anonymized_telemetry=False,
            allow_reset=True
        ))
        
        # Create or get collection
        self.collection_name = collection_name
        try:
            self.collection = self.client.get_collection(name=collection_name)
            # Reset collection if it exists (for session-scoped usage)
            self.client.delete_collection(name=collection_name)
        except:
            pass
        
        self.collection = self.client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Store original content for retrieval
        self.content_store: Dict[str, Dict[str, Any]] = {}
    
    def add_documents(self, summarized_items: List[Dict[str, Any]]):
        """
        Add summarized documents to the vector store.
        
        Args:
            summarized_items: List of content dictionaries with 'summary' field
        """
        ids = []
        documents = []
        embeddings = []
        metadatas = []
        
        for item in summarized_items:
            # Generate unique ID
            doc_id = str(uuid.uuid4())
            ids.append(doc_id)
            
            # Store the summary as the searchable document
            summary = item.get('summary', item.get('content', ''))
            documents.append(summary)
            
            # Generate embedding
            embedding = self.embedding_model.encode(summary).tolist()
            embeddings.append(embedding)
            
            # Store metadata
            metadata = {
                'page': item['page'],
                'type': item['type'],
                'doc_id': doc_id
            }
            metadatas.append(metadata)
            
            # Store original content for later retrieval
            self.content_store[doc_id] = {
                'original_content': item.get('content'),
                'summary': summary,
                'page': item['page'],
                'type': item['type'],
                'metadata': item.get('metadata', {})
            }
        
        # Add to collection
        if ids:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
    
    def search(self, query: str, n_results: int = 10) -> List[Dict[str, Any]]:
        """
        Perform dense vector search.
        
        Args:
            query: Search query string
            n_results: Number of results to return
        
        Returns:
            List of result dictionaries with 'id', 'distance', 'metadata', 'document'
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query).tolist()
        
        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        # Format results
        formatted_results = []
        if results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                doc_id = results['ids'][0][i]
                formatted_results.append({
                    'id': doc_id,
                    'distance': results['distances'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'document': results['documents'][0][i],
                    'original_content': self.content_store.get(doc_id, {}).get('original_content'),
                    'page': results['metadatas'][0][i].get('page'),
                    'type': results['metadatas'][0][i].get('type')
                })
        
        return formatted_results
    
    def get_original_content(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve original content by document ID.
        
        Args:
            doc_id: Document ID
        
        Returns:
            Original content dictionary or None
        """
        return self.content_store.get(doc_id)
    
    def reset(self):
        """Reset the vector store (clear all data)."""
        try:
            self.client.delete_collection(name=self.collection_name)
        except:
            pass
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.content_store = {}
    
    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Get all documents in the store with doc_id included."""
        documents = []
        for doc_id, content in self.content_store.items():
            doc = content.copy()
            doc['doc_id'] = doc_id
            documents.append(doc)
        return documents

