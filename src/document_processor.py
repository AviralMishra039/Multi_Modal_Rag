"""
DocumentProcessor: Handles PDF parsing for text, tables, and images.
"""

import pdfplumber
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
from typing import List, Dict, Any
import os


class DocumentProcessor:
    """Processes PDF documents to extract text, tables, and images."""
    
    def __init__(self, pdf_path: str):
        """
        Initialize the document processor.
        
        Args:
            pdf_path: Path to the PDF file
        """
        self.pdf_path = pdf_path
        self.text_blocks: List[Dict[str, Any]] = []
        self.tables: List[Dict[str, Any]] = []
        self.images: List[Dict[str, Any]] = []
    
    def process(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Process the PDF and extract all content.
        
        Returns:
            Dictionary with 'text', 'tables', and 'images' keys
        """
        # Open PDF with PyMuPDF for image extraction
        with fitz.open(self.pdf_path) as pdf_fitz:
            with pdfplumber.open(self.pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    # Extract text
                    text = page.extract_text()
                    if text and text.strip():
                        self.text_blocks.append({
                            'content': text,
                            'page': page_num,
                            'type': 'text',
                            'metadata': {}
                        })
                    
                    # Extract tables
                    tables = page.extract_tables()
                    for table_idx, table in enumerate(tables):
                        if table:
                            table_markdown = self._table_to_markdown(table)
                            self.tables.append({
                                'content': table_markdown,
                                'page': page_num,
                                'type': 'table',
                                'table_index': table_idx,
                                'metadata': {}
                            })
                    
                    # Extract images using PyMuPDF
                    if page_num <= len(pdf_fitz):
                        page_fitz = pdf_fitz[page_num - 1]
                        image_list = page_fitz.get_images()
                        for img_idx, img in enumerate(image_list):
                            image_dict = self._extract_image_fitz(pdf_fitz, img, page_num, img_idx)
                            if image_dict:
                                self.images.append(image_dict)
        
        return {
            'text': self.text_blocks,
            'tables': self.tables,
            'images': self.images
        }
    
    def _table_to_markdown(self, table: List[List[str]]) -> str:
        """Convert a table to markdown format."""
        if not table:
            return ""
        
        # Clean table data
        cleaned_table = []
        for row in table:
            cleaned_row = [str(cell) if cell is not None else "" for cell in row]
            cleaned_table.append(cleaned_row)
        
        if len(cleaned_table) == 0:
            return ""
        
        # Create markdown table
        markdown_lines = []
        header = cleaned_table[0]
        markdown_lines.append("| " + " | ".join(header) + " |")
        markdown_lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        
        for row in cleaned_table[1:]:
            # Pad row if needed
            while len(row) < len(header):
                row.append("")
            markdown_lines.append("| " + " | ".join(row[:len(header)]) + " |")
        
        return "\n".join(markdown_lines)
    
    def _extract_image_fitz(self, pdf_fitz, img_info: tuple, page_num: int, img_idx: int) -> Dict[str, Any]:
        """Extract image from PDF page using PyMuPDF."""
        try:
            # img_info is a tuple: (xref, smask, width, height, bpc, colorspace, alt. colorspace, name, filter, referencer)
            xref = img_info[0]
            
            # Extract image
            base_image = pdf_fitz.extract_image(xref)
            image_bytes = base_image["image"]
            
            return {
                'content': image_bytes,
                'page': page_num,
                'type': 'image',
                'image_index': img_idx,
                'metadata': {}
            }
        except Exception as e:
            print(f"Error extracting image from page {page_num}: {e}")
            return None
    
    def get_all_content(self) -> List[Dict[str, Any]]:
        """Get all extracted content as a flat list."""
        all_content = []
        all_content.extend(self.text_blocks)
        all_content.extend(self.tables)
        all_content.extend(self.images)
        return all_content

