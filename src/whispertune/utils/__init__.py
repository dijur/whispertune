from pathlib import Path
from typing import Dict, Any
import json
import yaml


def get_project_root() -> Path:
    """Get the root directory of the project."""
    return Path(__file__).parent.parent.parent.parent


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_metadata(metadata_path: str) -> Dict[str, str]:
    """Load audio metadata from a JSONL file."""
    metadata = {}
    with open(metadata_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            metadata[entry['key']] = entry['text']
    return metadata


def ensure_dir(path: Path) -> Path:
    """Ensure a directory exists, creating it if necessary."""
    path.mkdir(parents=True, exist_ok=True)
    return path
