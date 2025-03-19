from pathlib import Path
import torch
import torchaudio
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from typing import Dict, List, Optional
from tqdm import tqdm
import json
import logging
from src.utils.metrics import ASRMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WhisperEvaluator:
    def __init__(self, model_path: str, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = WhisperForConditionalGeneration.from_pretrained(model_path).to(self.device)
        self.processor = WhisperProcessor.from_pretrained(model_path)
        self.metrics = ASRMetrics()

    def transcribe(self, audio_path: str) -> str:
        """Transcribe a single audio file."""
        try:
            # Load and preprocess audio
            audio, sr = torchaudio.load(audio_path)
            if sr != 16000:
                audio = torchaudio.functional.resample(audio, sr, 16000)
            
            # Process through Whisper
            input_features = self.processor(
                audio.squeeze().numpy(),
                sampling_rate=16000,
                return_tensors="pt"
            ).input_features.to(self.device)
            
            # Generate transcription
            predicted_ids = self.model.generate(input_features)
            transcription = self.processor.batch_decode(
                predicted_ids, 
                skip_special_tokens=True
            )[0]
            
            return transcription.strip()
            
        except Exception as e:
            logger.error(f"Error transcribing {audio_path}: {e}")
            return ""

    def evaluate_dir(self, test_dir: str, metadata_path: Optional[str] = None) -> Dict[str, float]:
        """Evaluate model performance on a directory of test files."""
        test_dir = Path(test_dir)
        audio_files = list(test_dir.glob("*.wav"))
        
        if not audio_files:
            raise ValueError(f"No audio files found in {test_dir}")
        
        # Load reference texts if metadata file is provided
        references = {}
        if metadata_path:
            with open(metadata_path, 'r') as f:
                for line in f:
                    entry = json.loads(line)
                    references[Path(entry['audio_path']).name] = entry['text']
        
        # Process each audio file
        predictions = []
        ground_truth = []
        
        for audio_file in tqdm(audio_files, desc="Evaluating"):
            # Get transcription
            transcription = self.transcribe(str(audio_file))
            predictions.append(transcription)
            
            # Get reference text if available
            if metadata_path:
                ref_text = references.get(audio_file.name, "")
                ground_truth.append(ref_text)
        
        # Calculate metrics
        results = {}
        if ground_truth:
            results["wer"] = self.metrics.calculate_wer(predictions, ground_truth)
            results["cer"] = self.metrics.calculate_cer(predictions, ground_truth)
            results["bleu"] = self.metrics.calculate_bleu(predictions, ground_truth)
        
        # Save detailed results
        output_path = test_dir / "evaluation_results.jsonl"
        with open(output_path, 'w') as f:
            for i, (pred, audio_file) in enumerate(zip(predictions, audio_files)):
                result = {
                    "audio_file": str(audio_file),
                    "prediction": pred,
                }
                if ground_truth:
                    result["reference"] = ground_truth[i]
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        
        return results

    def evaluate_model(self, test_audio_dir: str, metadata_path: Optional[str] = None) -> Dict[str, float]:
        """Evaluate model performance and return metrics."""
        try:
            results = self.evaluate_dir(test_audio_dir, metadata_path)
            
            logger.info("\nEvaluation Results:")
            for metric, value in results.items():
                logger.info(f"{metric.upper()}: {value:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            return {}