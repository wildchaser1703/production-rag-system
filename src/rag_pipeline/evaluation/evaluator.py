from typing import Any, cast

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevance,
    context_precision,
    context_recall,
    faithfulness,
)

from rag_pipeline.utils.logger import log


class RAGEvaluator:
    """
    Evaluates the RAG pipeline using RAGAS metrics.
    """
    
    def __init__(self) -> None:
        self.metrics = [
            faithfulness,
            answer_relevance,
            context_precision,
            context_recall,
        ]

    def evaluate_pipeline(
        self, 
        questions: list[str], 
        answers: list[str], 
        contexts: list[list[str]], 
        ground_truths: list[str]
    ) -> dict[str, Any]:
        """
        Runs RAGAS evaluation on the provided data.
        """
        log.info(f"Evaluating pipeline on {len(questions)} samples")
        
        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths
        }
        
        dataset = Dataset.from_dict(data)
        
        try:
            results = evaluate(
                dataset,
                metrics=self.metrics,
            )
            log.success("Evaluation completed successfully")
            return cast(dict[str, Any], results)
        except Exception as e:
            log.error(f"Evaluation failed: {str(e)}")
            return {}
