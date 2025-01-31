import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from data.preprocessing import load_and_patch_data
from data.augmentation import get_train_augs, get_valid_augs
from data.dataset import SegmentationDataset
from models.segnet import ImageSegmentation
from models.loss import PartialFocalCrossEntropyLoss
from utils.metrics import calculate_accuracy
from config import Config

image_dataset, mask_dataset = load_and_patch_data(Config.ROOT_DIR, Config.PATCH_SIZE)

X_train, X_test, y_train, y_test = train_test_split(image_dataset, mask_dataset, test_size=0.20, random_state=42)

train_dataset = SegmentationDataset(X_train, y_train, get_train_augs())
test_dataset = SegmentationDataset(X_test, y_test, get_valid_augs())

train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE)

model = ImageSegmentation(kernel_size=3).to(Config.DEVICE)
criterion = PartialFocalCrossEntropyLoss(ignore_index=-1, alpha=1.0, gamma=2.0)
optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

for epoch in range(Config.NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    for images, masks in train_loader:
        images = images.float().to(Config.DEVICE)
        masks = masks.long().to(Config.DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{Config.NUM_EPOCHS}, Loss: {running_loss / len(train_loader)}")

torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)