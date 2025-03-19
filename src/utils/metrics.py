import nltk
from typing import List, Dict
import numpy as np
from jiwer import wer, cer
from nltk.translate.bleu_score import sentence_bleu

class ASRMetrics:
    @staticmethod
    def calculate_wer(references: List[str], hypotheses: List[str]) -> float:
        """Calculate Word Error Rate."""
        return wer(references, hypotheses)

    @staticmethod
    def calculate_cer(references: List[str], hypotheses: List[str]) -> float:
        """Calculate Character Error Rate."""
        return cer(references, hypotheses)

    @staticmethod
    def calculate_bleu(references: List[str], hypotheses: List[str]) -> float:
        """Calculate BLEU score."""
        ref_tokens = [[r.split()] for r in references]
        hyp_tokens = [h.split() for h in hypotheses]
        return np.mean([sentence_bleu([ref], hyp) for ref, hyp in zip(ref_tokens, hyp_tokens)])

    @staticmethod
    def print_metrics(references: List[str], hypotheses: List[str]) -> Dict[str, float]:
        """Calculate and print all metrics."""
        metrics = {
            "WER": ASRMetrics.calculate_wer(references, hypotheses),
            "CER": ASRMetrics.calculate_cer(references, hypotheses),
            "BLEU": ASRMetrics.calculate_bleu(references, hypotheses)
        }
        
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")
        
        return metrics
