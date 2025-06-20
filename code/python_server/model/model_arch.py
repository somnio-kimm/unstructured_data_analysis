# model_arch.py
# Store model architecture

# Library
import torch
from torch import nn
from torchvision import models

# Emotion classifier
class EmotionClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        # 사전 학습된 resnet18 로드
        base_model = models.resnet18(pretrained=True)

        # 마지막 FC 레이어 제거
        self.features = nn.Sequential(*list(base_model.children())[:-1])  # (batch, 512, 1, 1)

        # 새 출력 레이어 추가
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x