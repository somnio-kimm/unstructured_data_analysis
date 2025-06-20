# config.py
# Store constants

# Library
import torch

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_WEIGHT_PATH = "model/emotion_classifier5.pth"
EMOTION = ["anger", "happy", "panic", "sadness"]
EMOTION_MAP = {"anger": "sad",
			   "happy": "happy",
			   "panic": "sad",
			   "sadness": "sad"}
IMAGE_SIZE = 224