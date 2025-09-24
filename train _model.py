import os #Used to work with file paths (check if model file exists, etc.).
import torch #main PyTorch library for dl
import torch.nn as nn #Neural network module from PyTorch
import torch.optim as optim #optimizer like adam
from torchvision import datasets, transforms #datasets and transforms for image processing
from torch.utils.data import DataLoader #DataLoader to load images in batches
from PIL import Image #open iamges for prediction

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" Using device: {device}")

DATA_DIR = "asl_alphabet_train"   # Folder containing asl dataset
BATCH_SIZE = 32 # Batch size for training
EPOCHS = 10
MODEL_PATH = "model.pth" # Path to save the trained model
CLASSES_FILE = "classes.txt"#A text file to store all ASL class names for later.

transform = transforms.Compose([
    transforms.Resize((64, 64)), #bring evry image to 64x64
    transforms.ToTensor(), # Converts images to PyTorch tensors ([0,1] range).
    transforms.Normalize([0.5], [0.5]) # Normalize images to [-1, 1] range.
])