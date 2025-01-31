import os

class Config:
    PATCH_SIZE = 256
    BATCH_SIZE = 1
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"