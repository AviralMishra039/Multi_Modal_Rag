"""
AnswerGenerator: Generates grounded answers using retrieved context.
"""

import google.generativeai as genai
from typing import List, Dict, Any


class AnswerGenerator:
    """Generates answers from retrieved context using Gemini."""
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        """
        Initialize the answer generator.
        
        Args:
            api_key: Gemini API key
            model_name: Model name to use (default: gemini-2.5-flash)
        """
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
    
    def _format_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        Format retrieved documents into context string.
        
        Args:
            retrieved_docs: List of retrieved document dictionaries
        
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, doc in enumerate(retrieved_docs, start=1):
            content_type = doc.get('type', 'unknown')
            page_num = doc.get('page', '?')
            original_content = doc.get('original_content')
            
            if content_type == 'text':
                context_parts.append(f"[Source {i} - Page {page_num}, Text]\n{original_content}")
            elif content_type == 'table':
                context_parts.append(f"[Source {i} - Page {page_num}, Table]\n{original_content}")
            elif content_type == 'image':
                context_parts.append(f"[Source {i} - Page {page_num}, Image]\n(Diagram/image description available - see source details)")
            else:
                context_parts.append(f"[Source {i} - Page {page_num}]\n{original_content}")
            
            context_parts.append("")  # Add spacing
        
        return "\n".join(context_parts)
    
    def generate_answer(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate an answer from the query and retrieved context.
        
        Args:
            query: User's question
            retrieved_docs: List of retrieved document dictionaries
        
        Returns:
            Dictionary with 'answer' and 'sources' keys
        """
        if not retrieved_docs:
            return {
                'answer': "I couldn't find any relevant information to answer your question.",
                'sources': []
            }
        
        # Format context
        context = self._format_context(retrieved_docs)
        
        # Create prompt
        prompt = f"""You are a helpful assistant that answers questions based ONLY on the provided context from a document.

IMPORTANT RULES:
1. Answer ONLY using information from the provided context
2. If the context doesn't contain enough information to answer, say so explicitly
3. Do NOT make up or infer information that isn't in the context
4. Be concise but complete
5. Reference the page numbers when mentioning specific information

Context from document:
{context}

Question: {query}

Answer (based only on the provided context):"""

        try:
            response = self.model.generate_content(prompt)
            answer = response.text.strip()
            
            # Format sources
            sources = []
            for doc in retrieved_docs:
                sources.append({
                    'page': doc.get('page'),
                    'type': doc.get('type'),
                    'content_preview': self._get_content_preview(doc),
                    'original_content': doc.get('original_content')
                })
            
            return {
                'answer': answer,
                'sources': sources
            }
        except Exception as e:
            return {
                'answer': f"Error generating answer: {str(e)}",
                'sources': []
            }
    
    def _get_content_preview(self, doc: Dict[str, Any], max_length: int = 200) -> str:
        """Get a preview snippet of the content."""
        original_content = doc.get('original_content', '')
        content_type = doc.get('type', '')
        
        if content_type == 'image':
            return "[Image/Diagram]"
        elif content_type == 'table':
            # For tables, show first few lines
            lines = str(original_content).split('\n')[:3]
            preview = '\n'.join(lines)
            if len(str(original_content)) > len(preview):
                preview += "..."
            return preview
        else:  # text
            content_str = str(original_content)
            if len(content_str) > max_length:
                return content_str[:max_length] + "..."
            return content_str

