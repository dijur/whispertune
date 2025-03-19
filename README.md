<<<<<<< HEAD
# whispertune
A framework for fine-tuning OpenAI's Whisper model for low-resource languages using synthetic data generation
=======
# WhisperTune

A framework for fine-tuning OpenAI's Whisper model for low-resource languages using synthetic data generation.

## Overview

WhisperTune is a comprehensive toolkit designed to improve Automatic Speech Recognition (ASR) for low-resource languages through synthetic data generation and model fine-tuning. It addresses the challenge of limited audio data availability by leveraging text-to-speech technologies and providing robust evaluation metrics.

## Features

- Synthetic data generation pipeline for ASR training
- Fine-tuning framework for OpenAI's Whisper model
- Built-in evaluation metrics (WER, CER, BLEU)
- Support for low-resource languages
- Comprehensive data processing utilities

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/whispertune.git
cd whispertune
```

2. Install dependencies using Poetry:
```bash
poetry install
```

## Project Structure

- `src/data/` - Synthetic data generation and processing utilities
- `src/train/` - Fine-tuning pipeline for Whisper model
- `src/eval/` - Evaluation tools and metrics
- `config/` - Configuration files
- `notebooks/` - Jupyter notebooks for experimentation and analysis

## Usage

### Synthetic Data Generation

Use the `SyntheticDataGenerator` class to create training data:

```python
from src.data.synthetic_data import SyntheticDataGenerator

generator = SyntheticDataGenerator(config)
generator.process_text_corpus(
    input_file="path/to/text.txt",
    output_dir="path/to/output",
    json_output="metadata.json"
)
```

### Training

Fine-tune the Whisper model on your synthetic dataset:

```python
from src.train.trainer import WhisperTrainer

trainer = WhisperTrainer(config)
dataset = trainer.prepare_dataset(audio_paths, transcripts)
trainer.train(dataset)
```

### Evaluation

Evaluate model performance using built-in metrics:

```python
from src.utils.metrics import ASRMetrics

metrics = ASRMetrics.print_metrics(references, hypotheses)
```

## Configuration

The project uses YAML configuration files located in the `config/` directory:
- `config.yaml`: General configuration
- `experiments.yaml`: Experiment-specific settings

## Performance Metrics

The system provides several evaluation metrics:
- Word Error Rate (WER)
- Character Error Rate (CER)
- BLEU Score

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Add your license information here]

>>>>>>> 6b7d5d7 (Initial commit)
