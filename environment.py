# Mount Google Drive so checkpoints persist across free-Colab session restarts.
# If Drive is unavailable the notebook falls back to local /content storage.
import os

USE_DRIVE = True   # set False to skip Drive mount

if USE_DRIVE:
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        SAVE_DIR = '/content/drive/MyDrive/qm9_project'
    except Exception:
        USE_DRIVE = False

if not USE_DRIVE:
    SAVE_DIR = '/content/qm9_project'

os.makedirs(SAVE_DIR, exist_ok=True)
CKPT_DIR = os.path.join(SAVE_DIR, 'checkpoints')
os.makedirs(CKPT_DIR, exist_ok=True)
print('Save directory:', SAVE_DIR)

# Install dependencies (run once per Colab session).
import subprocess, sys

def pip(pkg):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg, '-q'])

pip('torch-geometric')
pip('rigl-torch')
print('All packages installed.')

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
import numpy as np
import matplotlib.pyplot as plt
import json, os

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', DEVICE)
torch.manual_seed(42)
np.random.seed(42)
from torch.utils.tensorboard import SummaryWriter