import os

DEFAULT_MODEL = os.environ.get("TQ_MODEL", "mlx-community/gemma-4-31b-it-4bit")
DEFAULT_CODEBOOK_DIMS = tuple(
    int(dim.strip())
    for dim in os.environ.get("TQ_CODEBOOK_DIMS", "256,512").split(",")
    if dim.strip()
)
