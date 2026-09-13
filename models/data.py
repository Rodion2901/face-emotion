import torch
import cv2
import numpy as np
import os
from torch.utils.data import DataLoader

PATH_EMOTIONS = "data/fer2013/test/"
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        self.sample = []
        for em in EMOTIONS:
            for img in os.listdir(PATH_EMOTIONS + em):
                self.sample.append([PATH_EMOTIONS + em + "/" + img, EMOTIONS.index(em)])
    def __len__(self):
        return len(self.sample)
    def __getitem__(self, idx):
        path, label = self.sample[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (48,48))
        img = img.astype("float32") / 255
        tensor = torch.from_numpy(img).unsqueeze(0)
        return tensor, label

dataset = Dataset()
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)