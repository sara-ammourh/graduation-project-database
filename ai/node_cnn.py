import torch
import torch.nn as nn

INPUT_SIZE = (1, 28 + 6 * 2, 28 + 6 * 2)
OUTPUT_SIZE = 62


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


class NodeCNN(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, num_classes=62):
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
