from pathlib import Path

# path pointing to data files (used to get file-names, caching)
data_path = Path('/insert/path/to/data')
# path to location of .hdf5 cache file for data
cache_path = Path('/insert/path/for/cache.hdf5')
# path to directory where adversaries should be stored
adversary_path = Path('/insert/path/for/adversaries')

save_path = Path(__file__).parent.parent.parent / 'misc'
