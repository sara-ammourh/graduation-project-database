from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple

import cv2
import easyocr
import numpy as np
from ultralytics import YOLO


class CharType(Enum):
    DIGIT = ("digit", "0")
    UPPER = ("upper", "X")
    LOWER = ("lower", "x")

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

    def __init__(self, seg_path: str) -> None:
        self.seg_model = YOLO(seg_path)
        self.reader = easyocr.Reader(["en"], gpu=True)

    def _seg_image(self, img_path: str, conf=0.4, iou=0.7):
        return self.seg_model.predict(source=img_path, conf=conf, iou=iou)

    def _convert_to_grayscale(self, node):
        if node.ndim == 3:
            return cv2.cvtColor(node, cv2.COLOR_BGR2GRAY)
        return node.copy()

    def _crop_borders(self, node_gray):
        h, w = node_gray.shape
        crop_px = min(12, h // 4, w // 4)
        if h > 2 * crop_px and w > 2 * crop_px:
            return node_gray[crop_px : h - crop_px, crop_px : w - crop_px]
        return node_gray

    def _apply_threshold(self, node_gray):
        _, t = cv2.threshold(node_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return 255 - t

    def _resize_node(self, node_gray):
        h, w = node_gray.shape
        return cv2.resize(node_gray, (max(64, w * 2), max(64, h * 2)))

    def _preprocess_node(self, node):
        node = self._convert_to_grayscale(node)
        node = self._crop_borders(node)
        node = self._apply_threshold(node)
        return self._resize_node(node)

    def _read_text_strict(self, img):
        return self.reader.readtext(
            img,
            detail=0,
            paragraph=False,
            text_threshold=0.5,
            low_text=0.3,
            allowlist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
        )

    def _read_text_lenient(self, img):
        return self.reader.readtext(
            img,
            detail=0,
            paragraph=False,
            low_text=0.1,
            text_threshold=0.2,
            link_threshold=0.1,
            mag_ratio=2.0,
        )

    def _pick_single_char(self, text: str) -> str:
        if not text:
            return ""
        chars = [c for c in text if c.isalnum()]
        for pred in (str.isdigit, str.isupper, str.islower):
            for c in chars:
                if pred(c):
                    return c
        return chars[0] if chars else ""

    def _determine_dominant_type(self, chars):
        counts = {
            CharType.DIGIT: sum(c.isdigit() for c in chars if c),
            CharType.UPPER: sum(c.isupper() for c in chars if c),
            CharType.LOWER: sum(c.islower() for c in chars if c),
        }
        return max(counts.items(), key=lambda x: x[1])[0]

    def _build_char_map(self, dominant_type: CharType):
        m = {}
        for c in self.CONFUSIONS:
            target = c.get_by_type(dominant_type)
            for t in CharType:
                if t != dominant_type:
                    m[c.get_by_type(t)] = target
        return m

    def _force_single_char(self, text, dominant_type: CharType):
        char_map = self._build_char_map(dominant_type)
        c = self._pick_single_char(text)
        if not c:
            return dominant_type.fallback
        c = char_map.get(c, c)
        if dominant_type == CharType.UPPER:
            c = c.upper()
        elif dominant_type == CharType.LOWER:
            c = c.lower()
        if dominant_type == CharType.DIGIT and not c.isdigit():
            return dominant_type.fallback
        if dominant_type == CharType.UPPER and not c.isupper():
            return dominant_type.fallback
        if dominant_type == CharType.LOWER and not c.islower():
            return dominant_type.fallback
        return c

    def _fix_confusions(self, texts):
        single_chars = [self._pick_single_char(t) for t in texts]

        result = []
        for c in single_chars:
            result.append(c if c else None)

        valid = [c for c in result if c]
        if not valid:
            return [CharType.UPPER.fallback for _ in texts]

        dominant = self._determine_dominant_type(valid)

        fixed = [
            self._force_single_char(c, dominant) if c else dominant.fallback
            for c in result
        ]

        used = set()
        all_possible = list(
            "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        )
        final = []
        for c in fixed:
            if c not in used:
                final.append(c)
                used.add(c)
            else:
                # pick next unused char
                for a in all_possible:
                    if a not in used:
                        final.append(a)
                        used.add(a)
                        break
        return final

    def _predict_node(self, node):
        g = self._preprocess_node(node)
        rgb = cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)
        texts = self._read_text_strict(rgb) or self._read_text_lenient(rgb)
        return g, texts[0] if texts else ""

    def _extract_nodes(self, results):
        nodes = []
        edges = []
        for r in results:
            img = r.orig_img
            boxes = r.boxes.xyxy.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy().astype(int)
            names = r.names
            for box, cls in zip(boxes, classes):
                x1, y1, x2, y2 = map(int, box)
                if names[cls] == "node":
                    crop = img[y1:y2, x1:x2]
                    gray, text = self._predict_node(crop)
                    nodes.append(
                        {
                            "center": ((x1 + x2) / 2, (y1 + y2) / 2),
                            "text": text,
                            "gray": gray,
                        }
                    )
                elif names[cls] == "edge":
                    edges.append({"bbox": (x1, y1, x2, y2)})
        return nodes, edges

    def _find_nearest_node(self, point, nodes):
        d, idx = float("inf"), -1
        for i, n in enumerate(nodes):
            dist = np.hypot(point[0] - n["center"][0], point[1] - n["center"][1])
            if dist < d:
                d, idx = dist, i
        return idx

    def _get_edge_corners(self, bbox):
        x1, y1, x2, y2 = bbox
        return [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]

    def _connect_edges_to_nodes(self, nodes, edges):
        graph = [GraphNode(i, n["text"], n["center"], []) for i, n in enumerate(nodes)]
        for e in edges:
            idxs = set(
                self._find_nearest_node(c, nodes)
                for c in self._get_edge_corners(e["bbox"])
            )
            if len(idxs) == 2:
                a, b = idxs
                if b not in graph[a].neighbors:
                    graph[a].neighbors.append(b)
                if a not in graph[b].neighbors:
                    graph[b].neighbors.append(a)
        return graph

    def predict_image(self, img_path):
        results = self._seg_image(img_path)
        nodes, edges = self._extract_nodes(results)
        texts = [n["text"] for n in nodes]
        fixed = self._fix_confusions(texts)
        for n, t in zip(nodes, fixed):
            n["text"] = t
        graph = self._connect_edges_to_nodes(nodes, edges)
        for n in graph:
            print(f"Node {n.id}: {n.label} → {n.neighbors}")
        return graph


if __name__ == "__main__":
    model = GraphModel("../../models/yolov8best.pt")
    graph = model.predict_image("test2.png")
    for n in graph:
        print(n.to_dict())
