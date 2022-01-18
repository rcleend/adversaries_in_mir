from pathlib import Path

# path pointing to data
# TODO: change to test data when testing
d_path = 'instrument_classifier/data/data_files/train_curated'
# path pointing to .csv files (containing label information)
csv_path = 'instrument_classifier/data/data_files/meta'

misc_path = str(Path(__file__).parent.parent.parent / 'misc')
adversary_path = str(Path(misc_path) / 'adversaries')
train_path = str(Path(misc_path) / 'pretrained_models')
model_path = str(Path(train_path) / 'models')
log_path = str(Path(train_path) / 'logs')
