from typing import cast

from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_pipeline.utils.logger import log


class DocumentProcessor:
    """
    Handles ingestion and chunking of documents from various sources.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

    def load_from_urls(self, urls: list[str]) -> list[Document]:
        """
        Loads document content from a list of URLs.
        """
        log.info(f"Loading content from {len(urls)} URLs")
        try:
            loader = WebBaseLoader(urls)
            documents = loader.load()
            log.success(f"Successfully loaded {len(documents)} documents")
            return documents
        except Exception as e:
            log.error(f"Failed to load documents: {str(e)}")
            return []

    def split_documents(self, documents: list[Document]) -> list[Document]:
        """
        Splits documents into smaller chunks for vector storage.
        """
        log.info("Splitting documents into chunks")
        chunks = self.text_splitter.split_documents(documents)
        log.success(f"Created {len(chunks)} chunks")
        return chunks  # type: ignore[no-any-return]
