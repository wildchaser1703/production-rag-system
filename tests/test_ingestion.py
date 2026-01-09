from langchain_core.documents import Document

from rag_pipeline.ingestion.loader import DocumentProcessor


def test_split_documents():
    """
    Test that the DocumentProcessor correctly splits documents into chunks.
    """
    chunk_size = 50
    processor = DocumentProcessor(chunk_size=chunk_size, chunk_overlap=10)
    content = (
        "This is a long sentence that should definitely be split "
        "into multiple chunks for testing purposes."
    )
    docs = [Document(page_content=content)]
    
    chunks = processor.split_documents(docs)
    
    assert len(chunks) > 1
    assert all(isinstance(c, Document) for c in chunks)
    assert len(chunks[0].page_content) <= chunk_size

def test_processor_initialization():
    """
    Test initialization with default values.
    """
    default_chunk_size = 1000
    default_overlap = 200
    processor = DocumentProcessor()
    assert processor.text_splitter._chunk_size == default_chunk_size
    assert processor.text_splitter._chunk_overlap == default_overlap
