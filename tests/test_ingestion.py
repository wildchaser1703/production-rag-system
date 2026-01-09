import pytest
from langchain_core.documents import Document
from src.rag_pipeline.ingestion.loader import DocumentProcessor

def test_split_documents():
    """
    Test that the DocumentProcessor correctly splits documents into chunks.
    """
    processor = DocumentProcessor(chunk_size=50, chunk_overlap=10)
    docs = [Document(page_content="This is a long sentence that should definitely be split into multiple chunks for testing purposes.")]
    
    chunks = processor.split_documents(docs)
    
    assert len(chunks) > 1
    assert all(isinstance(c, Document) for c in chunks)
    assert len(chunks[0].page_content) <= 50

def test_processor_initialization():
    """
    Test initialization with default values.
    """
    processor = DocumentProcessor()
    assert processor.text_splitter._chunk_size == 1000
    assert processor.text_splitter._chunk_overlap == 200
