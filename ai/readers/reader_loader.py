from enum import Enum
import os

from .base_reader import BaseReader
from .cnn_reader import CNNReader
from .ocr_reader import OCRReader

BACKEND_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


class ReaderType(Enum):
    CNN = (0, os.path.join(BACKEND_DIR, "models", "nodecnn_synthetic.pth"))
    OCR = (1, "")

    def __init__(self, _id, model_path) -> None:
        super().__init__()
        self._id = _id
        self.model_path = model_path

    @classmethod
    def from_str(cls, name: str) -> "ReaderType":
        normalized = (name or "cnn").strip().lower()
        for reader_type in cls:
            if reader_type.name.lower() == normalized:
                return reader_type
        raise ValueError(f"Invalid reader_type: {name}")


def load_reader(reader_type: ReaderType) -> BaseReader:
    if reader_type == ReaderType.CNN:
        return CNNReader(reader_type.model_path)
    elif reader_type == ReaderType.OCR:
        return OCRReader()
    else:
        raise ValueError(f"Invalid reader_type: {reader_type}")
