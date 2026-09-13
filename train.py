from src.emotion_detector import EmotionCNN
import torch.optim as optim
import torch.nn as nn
from models.data import dataloader
model = EmotionCNN()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

num_epoch = 50
for epoch in range(num_epoch):
    for images, lables in dataloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs,lables)
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0:
        print(loss.item())

torch.save(model.state_dict(), "models/emotion_model.pt")
