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


def subset(src, dst):
    exist_mkdir(dst)
    exist_mkdir(os.path.join(dst, 'train'))
    exist_mkdir(os.path.join(dst, 'val'))
    train_rate = 0.75
    class_names = os.listdir(src)
    class_names.sort()
    random.seed(1)
    for class_name in class_names:
        file_names = glob.glob(os.path.join(src, class_name, '*.png'))
        img_names = fnmatch.filter(file_names, '*.png')
        class_length = len(img_names)
        if class_length <= 4:
            print('passed ', class_name)
            continue
        else:
            print(class_name)
            random.shuffle(img_names)
            exist_mkdir(os.path.join(dst, 'train', class_name))
            exist_mkdir(os.path.join(dst, 'val', class_name))
            for i, img_name in enumerate(img_names):
                img = Image.open(img_name)
                name = os.path.basename(img_name)
                if i < class_length * train_rate:
                    img.save(os.path.join(dst, 'train', class_name, name))
                else:
                    img.save(os.path.join(dst, 'val', class_name, name))
                img.close()


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
    baseurl = 'http://www.nurs.or.jp/~nagadomi/animeface-character-dataset/data/'
    compressd_file = 'animeface-character-dataset.zip'
    filename = 'animeface-character-dataset'
    raw_path = 'data'
    dataset_root = 'animeface-character-dataset/thumb'
    exist_mkdir(raw_path)
    download_file(baseurl, compressd_file, raw_path)
    decompress_file(compressd_file, raw_path)
    subset(os.path.join(raw_path, dataset_root), raw_path)
    shutil.rmtree(os.path.join(raw_path, filename))
    os.remove(os.path.join(raw_path, compressd_file))
