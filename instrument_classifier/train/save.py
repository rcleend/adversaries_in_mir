
# THIS IS OUR OWN -OUTDATED- SAVING CODE. SEE TRAINING FOR THE CODE OF THE PAPER!

#
# import os
# import torch
# from instrument_classifier.utils.paths import model_path
#
#
# def save_model(nn, EPOCH):
#     model_name = "model_ep" + str(EPOCH)
#     torch.save({
#         'epoch': EPOCH,
#         'model_state_dict': nn.state_dict(),},
#         os.path.join(model_path, 'self_trained_torch/', model_name + '.tar'))
