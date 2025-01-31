import torch

def calculate_accuracy(outputs, targets):
    _, preds = torch.max(outputs, 1)
    correct_pixels = torch.sum(preds == targets)
    total_pixels = torch.numel(targets)
    accuracy = correct_pixels / total_pixels
    return accuracy.item()