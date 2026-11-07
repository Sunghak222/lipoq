"""VQA-specific selection for the matched random comparison."""
from common.archive import load_archive


def load_reference(source):
    return load_archive(source, method="vqa")
