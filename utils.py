# utils.py
import os
import json
from pathlib import Path


def load_jsonl(path, stream=False):
    path = Path(path)

    def _iter():
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    yield json.loads(line)

    return _iter() if stream else list(_iter())