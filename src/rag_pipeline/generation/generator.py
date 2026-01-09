from typing import Any, cast

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_openai import ChatOpenAI

from rag_pipeline.config import settings
from rag_pipeline.utils.logger import log


class RAGGenerator:
    """
    Handles the generation part of the RAG pipeline.
    """
    
    def __init__(self) -> None:
        self.llm = ChatOpenAI(
            model=settings.llm_model,
            temperature=0,
        )
        self.prompt = self._setup_prompt()

    def _setup_prompt(self) -> ChatPromptTemplate:
        """
        Defines the RAG system prompt.
        """
        template = """
        You are an advanced AI assistant for technical documentation. 
        Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know, 
        don't try to make up an answer.
        Use professional and technical language.

        Context:
        {context}

        Question: {question}

        Answer:
        """
        return ChatPromptTemplate.from_template(template)

    def get_chain(self, retriever: Any) -> Runnable[Any, Any]:
        """
        Builds the LangChain RAG chain.
        """
        log.info("Building RAG chain")
        
        def format_docs(docs: list[Any]) -> str:
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        
        return rag_chain

    def generate(self, query: str, retriever: Any) -> str:
        """
        Generates a response for a given query.
        """
        log.info(f"Generating response for query: {query}")
        chain = self.get_chain(retriever)
        response = chain.invoke(query)
        log.success("Response generated successfully")
        return cast(str, response)
