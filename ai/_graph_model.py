from dataclasses import dataclass, field
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from ultralytics import YOLO

from .node_cnn import INPUT_SIZE, OUTPUT_SIZE, NodeCNN, to_char


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
    def __init__(self, seg_path: str, cnn_path: str) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.seg_model = YOLO(seg_path)
        self.cnn_model = NodeCNN(input_size=INPUT_SIZE, num_classes=OUTPUT_SIZE)
        self.cnn_model.load_state_dict(torch.load(cnn_path, map_location=self.device))
        self.cnn_model.to(self.device)
        self.cnn_model.eval()

    def _seg_image(self, img_path, conf=0.4, iou=0.7):
        return self.seg_model.predict(img_path, conf=conf, iou=iou)

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

    def _predict_node(self, node_crop):
        processed = self.preprocess_node_emnist(node_crop)
        tensor = torch.from_numpy(processed).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.cnn_model(tensor)
            pred = logits.argmax(dim=1).item()

        return processed, to_char(pred)

    def _distance_to_mask(self, point, mask):
        if mask.ndim > 2:
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

        max_dist = img_height * 0.05
        graph = [GraphNode(i, n["text"], n["center"], []) for i, n in enumerate(nodes)]

        for e in edges:
            mask = e["mask"]
            distances = [
                (i, self._distance_to_mask(n["center"], mask))
                for i, n in enumerate(nodes)
            ]
            distances.sort(key=lambda x: x[1])

            if len(distances) >= 2:
                a_idx, a_dist = distances[0]
                b_idx, b_dist = distances[1]

                if a_dist <= max_dist and b_dist <= max_dist:
                    if b_idx not in graph[a_idx].neighbors:
                        graph[a_idx].neighbors.append(b_idx)
                    if a_idx not in graph[b_idx].neighbors:
                        graph[b_idx].neighbors.append(a_idx)
        return graph

    def predict_image(self, img_path: str, conf=0.4, iou=0.7, show_cnn_debug=False):
        results = self._seg_image(img_path, conf, iou)

        nodes = []
        edges = []
        debug_crops = []
        debug_texts = []

        for r in results:
            img = r.orig_img
            orig_h, orig_w = img.shape[:2]

            boxes = r.boxes.xyxy.cpu().numpy()  # pyright: ignore
            classes = r.boxes.cls.cpu().numpy().astype(int)  # pyright: ignore
            names = r.names

            masks_data = None
            if hasattr(r, "masks") and r.masks is not None:
                masks_data = r.masks.data

            for idx, (box, cls_id) in enumerate(zip(boxes, classes)):
                name = names[cls_id]
                x1, y1, x2, y2 = map(int, box)

                if name == "node":
                    crop = img[y1:y2, x1:x2]
                    processed_node, label_char = self._predict_node(crop)

                    nodes.append(
                        {"center": ((x1 + x2) / 2, (y1 + y2) / 2), "text": label_char}
                    )

                    if show_cnn_debug:
                        debug_crops.append(processed_node)
                        debug_texts.append(label_char)

                elif name == "edge" and masks_data is not None:
                    mask_tensor = masks_data[idx]
                    mask_np = mask_tensor.cpu().numpy()
                    mask_resized = cv2.resize(
                        mask_np, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
                    )
                    edges.append({"mask": mask_resized})

        if show_cnn_debug and debug_crops:
            n = len(debug_crops)
            cols = 5
            rows = (n + cols - 1) // cols
            plt.figure(figsize=(15, 3 * rows))
            for i, (crop, txt) in enumerate(zip(debug_crops, debug_texts)):
                plt.subplot(rows, cols, i + 1)
                plt.imshow(crop, cmap="gray")
                plt.title(f"Pred: {txt}")
                plt.axis("off")
            plt.tight_layout()
            plt.show()

        graph = self._connect_edges_to_nodes(nodes, edges, img_height=img.shape[0]) # pyright: ignore
        for n in graph:
            print(f"Node {n.id}: {n.label} at {n.pos} -> Neighbors: {n.neighbors}")

        return graph


if __name__ == "__main__":
    model = GraphModel(
        seg_path="../models/yolov8best.pt", cnn_path="../models/nodecnn_synthetic.pth"
    )
    graph = model.predict_image("test2.png", show_cnn_debug=True)
    for n in graph:
        print(n.to_dict())
