from rag_pipeline.generation.generator import RAGGenerator
from rag_pipeline.ingestion.loader import DocumentProcessor
from rag_pipeline.retrieval.vector_store import VectorManager
from rag_pipeline.utils.logger import log


def run_pipeline(urls: list[str], query: str) -> None:
    """
    Executes a full end-to-end RAG pipeline run.
    """
    # 1. Ingestion
    processor = DocumentProcessor()
    docs = processor.load_from_urls(urls)
    chunks = processor.split_documents(docs)
    
    # 2. Vector Storage
    vector_manager = VectorManager()
    vector_manager.initialize_store(chunks)
    retriever = vector_manager.get_retriever()
    
    # 3. Generation
    generator = RAGGenerator()
    answer = generator.generate(query, retriever)
    
    print("\n" + "="*50)
    print(f"QUERY: {query}")
    print(f"ANSWER: {answer}")
    print("="*50 + "\n")

if __name__ == "__main__":
    # Example usage: Answering about Python 3.12 features using official docs
    tech_urls = [
        "https://docs.python.org/3/whatsnew/3.12.html",
        "https://docs.python.org/3/whatsnew/3.11.html"
    ]
    sample_query = "What are the key performance improvements in Python 3.11 and 3.12?"
    
    try:
        run_pipeline(tech_urls, sample_query)
    except Exception as e:
        log.exception(f"Pipeline failed: {e}")
