import h5py

from recommender.utils.paths import data_path, cache_path
from recommender.data.soundpark_set import get_raw_soundpark_loader


def save_dataset(cache_path, data_loader):
    """ Saves all data to h5py-dataset located at defined path. """
    hf = h5py.File(cache_path, 'w')

    for b, (indices, middles) in enumerate(data_loader):
        print('File {}/{} is being stored...\r'.format(b + 1, len(data_loader)), flush=True, end='')
        hf.create_dataset(str(indices.item()), data=middles)

    hf.close()
    print('Successfully stored dataset!')


def main():
    # get file names
    clean_files = [str(f) for f in sorted(list(data_path.rglob('*.mp3')))]

    # get data (in loader)
    data_loader = get_raw_soundpark_loader(clean_files, 1)
    # save all files to h5py dataset
    save_dataset(cache_path, data_loader)


if __name__ == '__main__':
    main()
