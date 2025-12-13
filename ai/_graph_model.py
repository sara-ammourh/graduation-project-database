import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from node_cnn import INPUT_SIZE, OUTPUT_SIZE, NodeCNN, to_char
from ultralytics import YOLO


class GraphModel:
    def __init__(self, seg_path: str, cnn_path: str) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.seg_model = YOLO(seg_path)
        self.cnn_model = NodeCNN(input_size=INPUT_SIZE, num_classes=OUTPUT_SIZE)

        self.cnn_model.load_state_dict(torch.load(cnn_path, map_location=self.device))
        self.cnn_model.to(self.device)
        self.cnn_model.eval()

    def _seg_image(
        self,
        img_path: str,
        conf: float = 0.40,
        iou: float = 0.7,
        show: bool = False,
        save: bool = False,
    ) -> list:
        return self.seg_model.predict(
            source=img_path,
            conf=conf,
            iou=iou,
            show=show,
            save=save,
            project="output",
            name="graph_results",
        )

    @staticmethod
    def preprocess_node_emnist(node):
        if node.ndim == 3:
            node = cv2.cvtColor(node, cv2.COLOR_BGR2GRAY)

        h, w = node.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        # cv2.circle(mask, (w // 2, h // 2), min(h, w) // 2, 255, -1)
        node = cv2.bitwise_and(node, mask)

        _, node = cv2.threshold(node, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        node = 255 - node

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        node = cv2.erode(node, kernel, iterations=1)

        cnts, _ = cv2.findContours(node, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
            if len(cnts) > 1:
                symbol_mask = np.zeros_like(node)
                # cv2.drawContours(symbol_mask, cnts[1:], -1, 255, -1)
                node = symbol_mask

        node = cv2.dilate(node, kernel, iterations=1)

        ys, xs = np.where(node > 0)
        if len(xs) > 0 and len(ys) > 0:
            x1, x2 = xs.min(), xs.max()
            y1, y2 = ys.min(), ys.max()
            node = node[y1 : y2 + 1, x1 : x2 + 1]

        size = INPUT_SIZE[1]
        h, w = node.shape
        scale = (size - 4) / max(h, w)
        node = cv2.resize(node, (int(w * scale), int(h * scale)))

        canvas = np.zeros((size, size), dtype=np.uint8)
        y_off = (size - node.shape[0]) // 2
        x_off = (size - node.shape[1]) // 2
        canvas[y_off : y_off + node.shape[0], x_off : x_off + node.shape[1]] = node

        canvas = canvas.astype(np.float32) / 255.0
        return canvas

    def _predict_node(self, node):
        node = GraphModel.preprocess_node_emnist(node)
        tensor = torch.from_numpy(node).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.cnn_model(tensor)
            pred = logits.argmax(dim=1).item()

        return node, pred

    def predict_image(
        self,
        img_path: str,
        conf: float = 0.40,
        iou: float = 0.7,
        show: bool = False,
        save: bool = False,
    ) -> None:
        results = self._seg_image(img_path, conf=conf, iou=iou, show=show, save=save)
        all_nodes = []
        all_texts = []
        for r in results:
            img = r.orig_img
            boxes = r.boxes.xyxy.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy().astype(int)
            names = r.names

            for i, (box, cls_id) in enumerate(zip(boxes, classes)):
                if names[cls_id] != "node":
                    continue

                x1, y1, x2, y2 = map(int, box)
                node = img[y1:y2, x1:x2]
                new_node, label = self._predict_node(node)
                all_nodes.append(new_node)
                all_texts.append(to_char(label))
                print(f"Node {i}: {to_char(label)}")

        n = len(all_nodes)
        cols = 5
        rows = (n + cols - 1) // cols
        plt.figure(figsize=(15, 3 * rows))
        for i, (node, text) in enumerate(zip(all_nodes, all_texts)):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(node, cmap="gray")
            plt.title(text)
            plt.axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    image_path = "test2.png"

    graph_model = GraphModel(
        "../../models/yolov8best.pt",
        "../../models/nodecnn_synthetic.pth",
    )

    graph_model.predict_image(image_path)
