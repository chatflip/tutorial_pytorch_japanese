# -*- coding: utf-8 -*-
import fnmatch
import glob
import os
import random
import shutil
import sys
import urllib.request
import zipfile

from PIL import Image


def progress(block_count, block_size, total_size):
    percentage = min(int(100.0 * block_count * block_size / total_size), 100)
    bar = '[{}>{}]'.format('='*(percentage//4), ' '*(25-percentage//4))
    sys.stdout.write('{} {:3d}%\r'.format(bar, percentage))
    sys.stdout.flush()


def download_file(baseurl, filename, raw_path):
    if os.path.exists(os.path.join(raw_path, filename)):
        print('File exists: {}'.format(filename))
    else:
        print('Downloading: {}'.format(filename))
        try:
            urllib.request.urlretrieve(
                url=baseurl+filename,
                filename=os.path.join(raw_path, filename),
                reporthook=progress)
            print('')
        except (OSError, urllib.error.HTTPError) as err:
            print('ERROR :{}'.format(err.code))
            print(err.reason)


def decompress_file(filename, raw_path):
    with zipfile.ZipFile(os.path.join(raw_path, filename), 'r') as z:
        z.extractall(os.path.join(raw_path, ''))


def exist_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == '__main__':
    baseurl = 'https://www.cis.upenn.edu/~jshi/ped_html/'
    compressd_file = 'PennFudanPed.zip'
    raw_path = 'data'
    exist_mkdir(raw_path)
    download_file(baseurl, compressd_file, raw_path)
    decompress_file(compressd_file, raw_path)
    os.remove(os.path.join(raw_path, compressd_file))
