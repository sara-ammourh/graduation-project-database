from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

from .readers.reader_loader import ReaderType, load_reader


class CharType(Enum):
    DIGIT = ("digit", "0")
    UPPER = ("upper", "A")
    LOWER = ("lower", "a")

    def __init__(self, name, fallback):
        self._name = name
        self.fallback = fallback


@dataclass
class Confusion:
    digit: str
    upper: str
    lower: str

    def get_by_type(self, char_type: CharType) -> str:
        return {
            CharType.DIGIT: self.digit,
            CharType.UPPER: self.upper,
            CharType.LOWER: self.lower,
        }[char_type]


@dataclass
class GraphNode:
    id: int
    label: str
    pos: Tuple[float, float]
    neighbors: List[int] = field(default_factory=list)

    def to_dict(self):
        return {
            "id": self.id,
            "label": self.label,
            "pos": list(self.pos),
            "neighbors": self.neighbors,
        }


class GraphModel:
    CONFUSIONS = [
        Confusion("0", "O", "o"),
        Confusion("1", "I", "l"),
        Confusion("2", "Z", "z"),
        Confusion("5", "S", "s"),
        Confusion("8", "B", "b"),
        Confusion("6", "G", "g"),
    ]

    def __init__(self, seg_path: str, reader_type: ReaderType) -> None:
        self.seg_model = YOLO(seg_path)
        self.reader = load_reader(reader_type)

    def _seg_image(self, img_path, conf=0.2, iou=0.5):
        return self.seg_model.predict(img_path, conf=conf, iou=iou)

    def _preprocess_node(self, node):
        g = cv2.cvtColor(node, cv2.COLOR_BGR2GRAY)
        h, w = g.shape
        crop = min(10, h // 5, w // 5)
        if h > 2 * crop and w > 2 * crop:
            g = g[crop : h - crop, crop : w - crop]
        return cv2.resize(g, (max(64, g.shape[1] * 3), max(64, g.shape[0] * 3)))

    def _pick_single_char(self, text):
        for c in text:
            if c.isalnum():
                return c
        return ""

    def _determine_dominant_type(self, chars):
        scores = {
            CharType.DIGIT: sum(c.isdigit() for c in chars),
            CharType.UPPER: sum(c.isupper() for c in chars),
            CharType.LOWER: sum(c.islower() for c in chars),
        }
        total = sum(scores.values())
        if total == 0:
            return CharType.UPPER
        if scores[CharType.UPPER] + scores[CharType.LOWER] >= total * 0.4:
            return (
                CharType.UPPER
                if scores[CharType.UPPER] > scores[CharType.LOWER]
                else CharType.LOWER
            )
        return CharType.DIGIT

    def _build_char_map(self, dominant_type: CharType):
        char_map = {}
        for conf in self.CONFUSIONS:
            target = conf.get_by_type(dominant_type)
            for char_type in CharType:
                if char_type != dominant_type:
                    char_map[conf.get_by_type(char_type)] = target
        return char_map

    def _fix_confusions(self, texts):
        chars = [self._pick_single_char(t) for t in texts]
        valid = [c for c in chars if c]
        if not valid:
            return ["A"] * len(chars)

        dominant = self._determine_dominant_type(valid)
        char_map = self._build_char_map(dominant)

        char_pool = list("abcdefghijklmnopqrstuvwxyz")
        if dominant == CharType.DIGIT:
            char_pool = list("0123456789")
        elif dominant == CharType.UPPER:
            char_pool = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

        fixed = []
        for c in chars:
            if not c:
                fixed.append(dominant.fallback)
            else:
                mapped = char_map.get(c, c)
                if dominant == CharType.DIGIT:
                    if mapped.isdigit():
                        fixed.append(mapped)
                    else:
                        fixed.append(dominant.fallback)
                elif dominant == CharType.UPPER:
                    if mapped.isalpha():
                        fixed.append(mapped.upper())
                    else:
                        fixed.append(dominant.fallback)
                else:  # LOWER
                    if mapped.isalpha():
                        fixed.append(mapped.lower())
                    else:
                        fixed.append(dominant.fallback)

        used = set()
        final = []

        for c in fixed:
            if c not in used:
                final.append(c)
                used.add(c)
            else:
                for replacement in char_pool:
                    if replacement not in used:
                        final.append(replacement)
                        used.add(replacement)
                        break
                else:
                    final.append(c)

        return final

    def _distance_to_mask(self, point, mask):
        mask = np.squeeze(mask)
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return float("inf")
        dists = np.hypot(xs - point[0], ys - point[1])
        return np.min(dists)

    def _connect_edges_to_nodes(self, nodes, edges, img_height=None):
        if not nodes:
            return []
        if img_height is None:
            y_positions = [n["center"][1] for n in nodes]
            img_height = max(y_positions) * 2 if y_positions else 1000

        max_dist = max(img_height * 0.2, 50)
        graph = [GraphNode(i, n["text"], n["center"], []) for i, n in enumerate(nodes)]

        for e in edges:
            mask = e["mask"]
            distances = [
                (i, self._distance_to_mask(n["center"], mask))
                for i, n in enumerate(nodes)
            ]
            distances.sort(key=lambda x: x[1])
            if len(distances) >= 2:
                a, b = distances[0][0], distances[1][0]
                if distances[0][1] <= max_dist and distances[1][1] <= max_dist:
                    if b not in graph[a].neighbors:
                        graph[a].neighbors.append(b)
                    if a not in graph[b].neighbors:
                        graph[b].neighbors.append(a)
        return graph

    def display_ocr_predictions(self, img_path: str, conf=0.4, iou=0.7):
        results = self._seg_image(img_path, conf, iou)
        crops, raw_texts = [], []
        for r in results:
            img = r.orig_img
            boxes = r.boxes.xyxy.cpu().numpy()  # pyright: ignore
            classes = r.boxes.cls.cpu().numpy().astype(int)  # pyright: ignore
            names = r.names

            for box, cls in zip(boxes, classes):
                if names[cls] == "node":
                    x1, y1, x2, y2 = map(int, box)
                    crop = img[y1:y2, x1:x2]
                    g = self._preprocess_node(crop)
                    txt = self.reader.read_char(cv2.cvtColor(g, cv2.COLOR_GRAY2RGB))
                    crops.append(g)
                    raw_texts.append(txt)

        fixed = self._fix_confusions(raw_texts)
        if not crops:
            print("No nodes detected.")
            return
        cols = 5
        rows = (len(crops) + cols - 1) // cols
        plt.figure(figsize=(15, 3 * rows))

        for i, (crop, raw, fix) in enumerate(zip(crops, raw_texts, fixed)):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(crop, cmap="gray")
            plt.title(f"Raw: '{raw}' -> Fixed: '{fix}'", fontsize=10)
            plt.axis("off")
            print(f"Node {i}: Raw='{raw}' -> Fixed='{fix}'")

        plt.tight_layout()
        plt.show()

    def predict_image(self, img_path: str, conf=0.2, iou=0.5):
        results = self._seg_image(img_path, conf, iou)
        nodes, edges = [], []
        for r in results:
            img = r.orig_img
            boxes = r.boxes.xyxy.cpu().numpy()  # pyright: ignore
            classes = r.boxes.cls.cpu().numpy().astype(int)  # pyright: ignore
            confs = r.boxes.conf.cpu().numpy()  # pyright: ignore
            names = r.names
            masks = (
                r.masks.data.cpu().numpy()  # pyright: ignore
                if hasattr(r, "masks") and r.masks is not None
                else None
            )

            for idx, (box, cls, score) in enumerate(zip(boxes, classes, confs)):
                if names[cls] == "node":
                    if score < 0.8:
                        continue

                    x1, y1, x2, y2 = map(int, box)
                    crop = img[y1:y2, x1:x2]
                    g = self._preprocess_node(crop)
                    txt = self.reader.read_char(cv2.cvtColor(g, cv2.COLOR_GRAY2RGB))
                    print(f"predicted: {txt}")
                    nodes.append(
                        {"center": ((x1 + x2) / 2, (y1 + y2) / 2), "text": txt}
                    )
                elif names[cls] == "edge" and masks is not None:
                    mask = masks[idx].astype(np.uint8)
                    mask = cv2.resize(
                        mask,
                        (img.shape[1], img.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    edges.append({"mask": mask})

        texts = [n["text"] for n in nodes]
        fixed = self._fix_confusions(texts)
        for n, t in zip(nodes, fixed):
            n["text"] = t
        graph = self._connect_edges_to_nodes(nodes, edges)
        for n in graph:
            print(f"Node {n.id}: {n.label} at {n.pos} -> {n.neighbors}")
        return graph


if __name__ == "__main__":
    model = GraphModel("../../models/yolov8seg.pt", ReaderType.OCR)
    model.display_ocr_predictions("test2.png")
    graph = model.predict_image("test2.png")
    for n in graph:
        print(n.to_dict())
