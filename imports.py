import logging
from torch.utils.tensorboard import SummaryWriter
import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import argparse
import os
import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import random_split, DataLoader
from PIL import Image
import gdown
import zipfile
import os