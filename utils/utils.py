from os import makedirs, listdir
from os.path import exists

def path_exists(path):
    if not exists(path):
        makedirs(path)
    return path