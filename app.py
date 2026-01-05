"""
Streamlit UI for Multi-Modal RAG Explorer
"""

import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

from src.document_processor import DocumentProcessor
from src.summarizer import Summarizer
from src.vector_store import VectorStore
from src.retriever import Retriever
from src.answer_generator import AnswerGenerator

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Multi-Modal RAG Explorer",
    page_icon="📄",
    layout="wide"
)

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'document_processed' not in st.session_state:
    st.session_state.document_processed = False
if 'processed_content' not in st.session_state:
    st.session_state.processed_content = None

def process_document(pdf_file, api_key):
    """Process uploaded PDF document."""
    with st.spinner("Processing PDF document..."):
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(pdf_file.read())
            tmp_path = tmp_file.name
        
        try:
            # Step 1: Extract content
            st.info("Step 1/4: Extracting text, tables, and images from PDF...")
            processor = DocumentProcessor(tmp_path)
            extracted_content = processor.process()
            
            # Step 2: Generate summaries
            st.info("Step 2/4: Generating summaries for tables and images...")
            summarizer = Summarizer(api_key)
            all_content = processor.get_all_content()
            summarized_content = summarizer.summarize_batch(all_content)
            
            # Step 3: Index in vector store
            st.info("Step 3/4: Indexing content in vector store...")
            vector_store = VectorStore()
            vector_store.add_documents(summarized_content)
            
            # Step 4: Initialize retriever
            st.info("Step 4/4: Initializing retriever...")
            retriever = Retriever(vector_store)
            retriever.rebuild_index()
            
            # Store in session state
            st.session_state.vector_store = vector_store
            st.session_state.retriever = retriever
            st.session_state.document_processed = True
            st.session_state.processed_content = {
                'text_count': len(extracted_content['text']),
                'table_count': len(extracted_content['tables']),
                'image_count': len(extracted_content['images'])
            }
            
            st.success("Document processed successfully!")
            return True
            
        except Exception as e:
            st.error(f"Error processing document: {str(e)}")
            return False
        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_path)
            except:
                pass

def main():
    """Main Streamlit app."""
    st.title("📄 Multi-Modal RAG Explorer")
    st.markdown("Upload a PDF and ask questions about its content (text, tables, and images)")
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # API Key input
        api_key = st.text_input(
            "Gemini API Key",
            type="password",
            value=os.getenv("GEMINI_API_KEY", ""),
            help="Enter your Google Gemini API key"
        )
        
        st.divider()
        
        # PDF Upload
        st.header("Document Upload")
        uploaded_file = st.file_uploader(
            "Upload PDF",
            type=['pdf'],
            help="Upload a PDF document to query"
        )
        
        if uploaded_file is not None:
            if st.button("Process Document", type="primary"):
                if not api_key:
                    st.error("Please enter your Gemini API key")
                else:
                    process_document(uploaded_file, api_key)
        
        # Document status
        if st.session_state.document_processed and st.session_state.processed_content:
            st.divider()
            st.header("Document Status")
            st.success("✅ Document Processed")
            content = st.session_state.processed_content
            st.metric("Text Blocks", content['text_count'])
            st.metric("Tables", content['table_count'])
            st.metric("Images", content['image_count'])
            
            if st.button("Reset", type="secondary"):
                st.session_state.vector_store = None
                st.session_state.retriever = None
                st.session_state.document_processed = False
                st.session_state.processed_content = None
                st.rerun()
    
    # Main interface
    if not st.session_state.document_processed:
        st.info("👈 Please upload and process a PDF document to get started.")
    else:
        # Query interface
        st.header("Ask a Question")
        query = st.text_input(
            "Enter your question",
            placeholder="e.g., What are the key findings in the results table?",
            label_visibility="collapsed"
        )
        
        if st.button("Search", type="primary") and query:
            if not api_key:
                st.error("Please enter your Gemini API key in the sidebar")
            else:
                with st.spinner("Searching and generating answer..."):
                    try:
                        # Retrieve relevant documents
                        retrieved_docs = st.session_state.retriever.retrieve(query, top_k=5)
                        
                        # Generate answer
                        answer_generator = AnswerGenerator(api_key)
                        result = answer_generator.generate_answer(query, retrieved_docs)
                        
                        # Display answer
                        st.header("Answer")
                        st.write(result['answer'])
                        
                        # Display sources
                        st.header("Sources")
                        for idx, source in enumerate(result['sources'], start=1):
                            with st.expander(f"Source {idx} - Page {source['page']} ({source['type'].upper()})"):
                                if source['type'] == 'table':
                                    st.markdown("**Table Content:**")
                                    st.code(source['content_preview'], language='markdown')
                                    # Show full table if needed
                                    if st.checkbox(f"Show full table (Source {idx})", key=f"show_table_{idx}"):
                                        st.code(source['original_content'], language='markdown')
                                elif source['type'] == 'image':
                                    st.markdown("**Image/Diagram:**")
                                    st.info("Image content is available in the original document at the referenced page.")
                                    # Note: Could display image here if we store it differently
                                else:  # text
                                    st.markdown("**Text Content:**")
                                    st.text(source['content_preview'])
                                    if st.checkbox(f"Show full text (Source {idx})", key=f"show_text_{idx}"):
                                        st.text_area(
                                            "Full Content",
                                            value=source['original_content'],
                                            height=200,
                                            key=f"full_text_{idx}",
                                            disabled=True
                                        )
                        
                    except Exception as e:
                        st.error(f"Error generating answer: {str(e)}")
                        st.exception(e)

if __name__ == "__main__":
    main()

