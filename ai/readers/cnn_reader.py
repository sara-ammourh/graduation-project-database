from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn

from .base_reader import BaseReader

INPUT_SIZE = (1, 28 + 6 * 2, 28 + 6 * 2)
OUTPUT_SIZE = 62


class NodeCNN(
    nn.Module,
):
    def __init__(self, input_size=INPUT_SIZE, num_classes=OUTPUT_SIZE):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, *input_size)
            flattened_size = self.features(dummy).view(1, -1).size(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class CNNReader(BaseReader):
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = NodeCNN()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def preprocess_node_emnist(node):
        """Preprocessing: Gray -> Contrast -> Invert -> Dilate -> Resize -> Pad."""
        if node.ndim == 3:
            node = cv2.cvtColor(node, cv2.COLOR_BGR2GRAY)

        # Increase contrast to make the text stand out more before thresholding
        node = cv2.convertScaleAbs(node, alpha=1.5, beta=0)

        # Threshold to create a strict binary image
        _, node = cv2.threshold(node, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Invert: CNN expects white text on black background
        node = 255 - node

        # Dilate slightly to strengthen the white strokes, making them less "pale"
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        node = cv2.dilate(node, kernel, iterations=1)

        # Resize preserving aspect ratio
        size = INPUT_SIZE[1]
        h, w = node.shape
        scale = (size - 4) / max(h, w) if max(h, w) > 0 else 1
        node = cv2.resize(node, (int(w * scale), int(h * scale)))

        # Center on black canvas
        canvas = np.zeros((size, size), dtype=np.uint8)
        y_off = (size - node.shape[0]) // 2
        x_off = (size - node.shape[1]) // 2
        canvas[y_off : y_off + node.shape[0], x_off : x_off + node.shape[1]] = node

        # Normalize
        canvas = canvas.astype(np.float32) / 255.0
        return canvas

    @staticmethod
    def to_char(class_label: int):
        if 0 <= class_label < 10:
            return str(class_label)

        class_label -= 10
        if 0 <= class_label < 26:
            return chr(class_label + ord("a"))

        class_label -= 26
        if 0 <= class_label < 26:
            return chr(class_label + ord("A"))

        raise ValueError(f"Invalid class_label: {class_label}")

    def read_char(self, img: Any) -> str:
        processed = self.preprocess_node_emnist(img)
        tensor = torch.from_numpy(processed).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            pred = logits.argmax(dim=1).item()

        return CNNReader.to_char(pred)
