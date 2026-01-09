from typing import Any, cast

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import OpenAIEmbeddings

from rag_pipeline.config import settings
from rag_pipeline.utils.logger import log


class VectorManager:
    """
    Manages the lifecycle of the vector database.
    """
    
    def __init__(self) -> None:
        self.embeddings = OpenAIEmbeddings(
            model=settings.embedding_model
        )
        self.vector_store: Chroma | None = None

    def initialize_store(self, documents: list[Document] | None = None) -> None:
        """
        Initializes or loads the Chroma vector store.
        """
        log.info(f"Initializing vector store at {settings.vector_db_path}")
        
        if documents:
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=settings.vector_db_path,
                collection_name=settings.collection_name
            )
        else:
            self.vector_store = Chroma(
                persist_directory=settings.vector_db_path,
                embedding_function=self.embeddings,
                collection_name=settings.collection_name
            )
        
        log.success("Vector store initialized successfully")

    def get_retriever(
        self, search_kwargs: dict[str, Any] | None = None
    ) -> VectorStoreRetriever:
        """
        Returns a retriever interface for the vector store.
        """
        if not self.vector_store:
            self.initialize_store()
            
        if not self.vector_store:
            raise RuntimeError("Failed to initialize vector store")

        retriever = self.vector_store.as_retriever(
            search_kwargs=search_kwargs or {"k": 4}
        )
        return retriever  # type: ignore[no-any-return]
