"""
Summarizer: Generates LLM-based summaries for tables and images.
"""

import google.generativeai as genai
from typing import Dict, Any, List
import io
from PIL import Image


class Summarizer:
    """Generates semantic summaries for non-text content using LLM."""
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        """
        Initialize the summarizer.
        
        Args:
            api_key: Gemini API key
            model_name: Model name to use (default: gemini-2.5-flash)
        """
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
    
    def summarize_table(self, table_markdown: str, page_num: int) -> str:
        """
        Generate a summary for a table.
        
        Args:
            table_markdown: Markdown representation of the table
            page_num: Page number where the table appears
        
        Returns:
            Summary string describing the table
        """
        prompt = f"""Analyze the following table from page {page_num} of a document and provide a concise, structured summary.

Focus on:
- Key trends or patterns
- Important comparisons
- Notable values or statistics
- The overall purpose or meaning

Table (Markdown):
{table_markdown}

Provide a clear, technical summary (2-4 sentences):"""

        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"Error summarizing table on page {page_num}: {e}")
            return f"Table on page {page_num} (summary generation failed)"
    
    def summarize_image(self, image_bytes: bytes, page_num: int) -> str:
        """
        Generate a description for an image/diagram.
        
        Args:
            image_bytes: Image data as bytes (can be None)
            page_num: Page number where the image appears
        
        Returns:
            Description string of the image
        """
        try:
            if image_bytes is None:
                # If image extraction failed, provide a placeholder description
                return f"Image/diagram on page {page_num} (image content available in original document at this page)"
            
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_bytes))
            
            prompt = """Analyze this image from a technical document and provide a structured, technical description.

Focus on:
- Key components or elements visible
- Flow, structure, or relationships shown
- Labels, text, or annotations present
- Overall purpose and meaning in the document context

Provide a clear, technical description (3-5 sentences):"""

            response = self.model.generate_content([prompt, image])
            return response.text.strip()
        except Exception as e:
            print(f"Error summarizing image on page {page_num}: {e}")
            return f"Image on page {page_num} (description generation failed)"
    
    def summarize_batch(self, content_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate summaries for a batch of content items.
        
        Args:
            content_items: List of content dictionaries with 'type', 'content', 'page'
        
        Returns:
            List of content dictionaries with added 'summary' field
        """
        summarized_items = []
        
        for item in content_items:
            item_copy = item.copy()
            
            if item['type'] == 'table':
                item_copy['summary'] = self.summarize_table(
                    item['content'],
                    item['page']
                )
            elif item['type'] == 'image':
                item_copy['summary'] = self.summarize_image(
                    item['content'],
                    item['page']
                )
            else:  # text
                # For text, we use the content itself (no summarization needed)
                item_copy['summary'] = item['content']
            
            summarized_items.append(item_copy)
        
        return summarized_items

