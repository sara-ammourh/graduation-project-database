from typing import Any

import easyocr

from .base_reader import BaseReader


class OCRReader(BaseReader):
    def __init__(self):
        self.reader = easyocr.Reader(["en"], gpu=True)

    def read_char(self, img: Any) -> str:
        results = self.reader.readtext(
            img,
            detail=1,
            low_text=0.1,
            text_threshold=0.05,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
            paragraph=False,
        )
        if not results:
            return ""
        best = max(results, key=lambda x: x[2])  # pyright: ignore
        return best[1].strip()  # pyright: ignore
