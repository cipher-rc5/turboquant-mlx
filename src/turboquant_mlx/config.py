import os

DEFAULT_MODEL_PATH = "/Users/excalibur/.lmstudio/models/mlx-community/gemma-4-31b-it-4bit"

DEFAULT_MODEL = os.environ.get("TQ_MODEL", DEFAULT_MODEL_PATH)
