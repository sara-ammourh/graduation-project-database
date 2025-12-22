from enum import Enum

from .base_reader import BaseReader
from .cnn_reader import CNNReader
from .ocr_reader import OCRReader


class ReaderType(Enum):
    CNN = (0, "models/nodecnn_synthetic.pth")
    OCR = (1, "")


def load_reader(reader_type: ReaderType) -> BaseReader:
    if reader_type == ReaderType.CNN:
        return CNNReader(reader_type.value[1])
    elif reader_type == ReaderType.OCR:
        return OCRReader()
    else:
        raise ValueError(f"Invalid reader_type: {reader_type}")
