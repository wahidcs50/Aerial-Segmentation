import torch
from torch.utils.data import DataLoader
from data.dataset import SegmentationDataset
from data.augmentation import get_valid_augs
from models.segnet import ImageSegmentation
from models.loss import PartialFocalCrossEntropyLoss
from utils.metrics import calculate_accuracy
from utils.visualization import plot_single_example
from config import Config

model = ImageSegmentation(kernel_size=3).to(Config.DEVICE)
model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH))
model.eval()

test_dataset = SegmentationDataset(X_test, y_test, get_valid_augs())
test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE)

criterion = PartialFocalCrossEntropyLoss(ignore_index=-1, alpha=1.0, gamma=2.0)
running_loss = 0.0
with torch.no_grad():
    for images, masks in test_loader:
        images = images.float().to(Config.DEVICE)
        masks = masks.long().to(Config.DEVICE)
        outputs = model(images)
        loss = criterion(outputs, masks)
        running_loss += loss.item()
        accuracy = calculate_accuracy(outputs, masks)
        print(f"Loss: {loss.item()}, Accuracy: {accuracy}")

print(f"Average Test Loss: {running_loss / len(test_loader)}")

plot_single_example(images[0].cpu().numpy(), masks[0].cpu().numpy(), outputs.argmax(dim=1)[0].cpu().numpy())