import os #Used to work with file paths (check if model file exists, etc.).
import torch #main PyTorch library for dl
import torch.nn as nn #Neural network module from PyTorch
import torch.optim as optim #optimizer like adam
from torchvision import datasets, transforms #datasets and transforms for image processing
from torch.utils.data import DataLoader #DataLoader to load images in batches
from PIL import Image #open iamges for prediction

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" Using device: {device}")

