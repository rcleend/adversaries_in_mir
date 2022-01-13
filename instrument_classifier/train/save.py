import torch
from instrument_classifier.utils.paths import model_path

def save_model(net, model_name):
    torch.save(net.state_dict(), os.path.join(model_path, model_name), _use_new_zipfile_serialization=False)