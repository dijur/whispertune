import torch
import torchaudio
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from datasets import Dataset
from typing import List, Optional, Dict
from pathlib import Path
import logging
import wandb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WhisperTrainer:
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device(config['training']['device'])
        self.model = WhisperForConditionalGeneration.from_pretrained(
            config['whisper']['model_name']
        ).to(self.device)
        self.processor = WhisperProcessor.from_pretrained(
            config['whisper']['model_name']
        )
        
        # Initialize wandb if configured
        if config.get('wandb', {}).get('entity'):
            wandb.init(
                project=config['wandb']['project'],
                entity=config['wandb']['entity'],
                config=config
            )
    
    def prepare_dataset(self, audio_paths: List[str]) -> Dataset:
        """Prepare dataset for training."""
        logger.info(f"Preparing dataset from {len(audio_paths)} audio files")
        
        def process_audio(audio_path):
            """Load and preprocess audio file."""
            audio, sr = torchaudio.load(audio_path)
            if sr != self.config['data']['sample_rate']:
                audio = torchaudio.functional.resample(
                    audio, 
                    sr, 
                    self.config['data']['sample_rate']
                )
            return audio.squeeze().numpy()
        
        def load_metadata(audio_path: str) -> str:
            """Load corresponding text from metadata."""
            metadata_path = Path(audio_path).with_suffix('.txt')
            if metadata_path.exists():
                return metadata_path.read_text().strip()
            return ""
        
        features = []
        for audio_path in audio_paths:
            try:
                audio = process_audio(audio_path)
                text = load_metadata(audio_path)
                
                inputs = self.processor(
                    audio,
                    sampling_rate=self.config['data']['sample_rate'],
                    text=text,
                    return_tensors="pt",
                    padding=True
                )
                
                features.append({
                    'input_ids': inputs.input_ids.squeeze(),
                    'attention_mask': inputs.attention_mask.squeeze(),
                    'labels': inputs.labels.squeeze()
                })
                
            except Exception as e:
                logger.warning(f"Error processing {audio_path}: {e}")
                continue
                
        return Dataset.from_list(features)
    
    def compute_metrics(self, pred):
        """Compute training metrics."""
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        
        # Decode predictions and labels
        pred_str = self.processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.processor.batch_decode(label_ids, skip_special_tokens=True)
        
        # Calculate WER
        wer = self.calculate_wer(pred_str, label_str)
        
        return {"wer": wer}
    
    def calculate_wer(self, predictions: List[str], references: List[str]) -> float:
        """Calculate Word Error Rate."""
        total_wer = 0
        total_words = 0
        
        for pred, ref in zip(predictions, references):
            pred_words = pred.split()
            ref_words = ref.split()
            
            # Calculate Levenshtein distance
            distance = self.levenshtein_distance(pred_words, ref_words)
            total_wer += distance
            total_words += len(ref_words)
        
        return total_wer / total_words if total_words > 0 else 1.0
    
    @staticmethod
    def levenshtein_distance(s1: List[str], s2: List[str]) -> int:
        """Calculate Levenshtein distance between two word sequences."""
        if len(s1) < len(s2):
            return WhisperTrainer.levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def train(self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None):
        """Fine-tune the Whisper model."""
        logger.info("Starting training...")
        
        training_args = Seq2SeqTrainingArguments(
            output_dir=self.config['training']['output_dir'],
            learning_rate=self.config['whisper']['learning_rate'],
            num_train_epochs=self.config['whisper']['num_epochs'],
            per_device_train_batch_size=self.config['whisper']['batch_size'],
            gradient_accumulation_steps=self.config['whisper']['gradient_accumulation_steps'],
            warmup_steps=self.config['whisper']['warmup_steps'],
            logging_steps=self.config['training']['logging_steps'],
            evaluation_strategy="steps" if eval_dataset else "no",
            save_steps=self.config['training']['save_steps'],
            fp16=torch.cuda.is_available(),
            report_to="wandb" if self.config.get('wandb', {}).get('entity') else None
        )
        
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics
        )
        
        trainer.train()
        
        # Save the final model
        output_dir = Path(self.config['training']['output_dir']) / "final"
        trainer.save_model(output_dir)
        logger.info(f"Model saved to {output_dir}")
