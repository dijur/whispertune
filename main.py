import argparse
import yaml
from pathlib import Path
from src.data.synthetic_data import SyntheticDataGenerator
from src.train.trainer import WhisperTrainer
from src.utils.metrics import ASRMetrics

def load_config(config_path: str):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main(args):
    # Load configuration
    config = load_config(args.config)
    
    if args.generate:
        # Generate synthetic data
        data_generator = SyntheticDataGenerator(config)
        data_generator.process_text_corpus(
            input_file=Path(config['data']['text_corpus_path']) / args.input,
            output_dir=config['data']['synthetic_audio_path'],
            json_output=Path(config['data']['synthetic_audio_path']) / 'metadata.jsonl'
        )
    
    if args.train:
        # Initialize trainer
        trainer = WhisperTrainer(config)
        
        # Prepare dataset
        audio_paths = list(Path(config['data']['synthetic_audio_path']).glob('*.wav'))
        dataset = trainer.prepare_dataset(audio_paths)
        
        # Train model
        trainer.train(dataset)
    
    if args.evaluate:
        # Load test data and run evaluation
        metrics = ASRMetrics()
        results = metrics.evaluate_model(
            model_path=args.model_path,
            test_audio_dir=args.test_dir
        )
        print("\nEvaluation Results:")
        for metric, value in results.items():
            print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='WhisperTune - Fine-tune Whisper for low-resource languages')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config file')
    parser.add_argument('--generate', action='store_true', help='Generate synthetic data')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the model')
    parser.add_argument('--input', type=str, help='Input text file for synthetic data generation')
    parser.add_argument('--model_path', type=str, help='Path to trained model for evaluation')
    parser.add_argument('--test_dir', type=str, help='Directory containing test audio files')
    
    args = parser.parse_args()
    main(args)
