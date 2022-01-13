import torch
from instrument_classifier.utils.paths import model_path
from instrument_classifier.train.train import net

def save_model(nn, model_name):
    torch.save(nn.state_dict(), os.path.join(model_path, model_name), _use_new_zipfile_serialization=False)

save_model(net, "save_test")