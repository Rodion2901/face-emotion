import torch
import torch.nn as nn

class EmotionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.affine = nn.Sequential(
            nn.Conv2d(1,32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(32,64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(),
            nn.Conv2d(64,128, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(128,128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(),
            nn.Conv2d(128,128, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(128,128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(),
            nn.Flatten(),
            nn.Linear(512,32),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(32, 7)
        )
    def forward(self, x):
        x = self.affine(x)
        return x
