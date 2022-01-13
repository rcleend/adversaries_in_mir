import torch
import os
from instrument_classifier.utils.paths import model_path


def save_model(nn, model_name):
    torch.save(nn.state_dict(), os.path.join(model_path, "/self-trained", model_name), _use_new_zipfile_serialization=False)
