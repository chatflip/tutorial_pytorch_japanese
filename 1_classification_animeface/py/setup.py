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
    os.makedirs(dst, exist_ok=True)
    os.makedirs(os.path.join(dst, 'train'), exist_ok=True)
    os.makedirs(os.path.join(dst, 'val'), exist_ok=True)
    train_rate = 0.75
    class_names = os.listdir(src)
    class_names.sort()
    random.seed(1)
    num_classes = 0
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
            os.makedirs(os.path.join(dst, 'train', class_name), exist_ok=True)
            os.makedirs(os.path.join(dst, 'val', class_name), exist_ok=True)
            for i, img_name in enumerate(img_names):
                img = Image.open(img_name)
                name = os.path.basename(img_name)
                if i < class_length * train_rate:
                    img.save(os.path.join(dst, 'train', class_name, name))
                else:
                    img.save(os.path.join(dst, 'val', class_name, name))
                img.close()
            num_classes += 1
    print('num of classes: {}'.format(num_classes))


def progress(block_count, block_size, total_size):
    percentage = min(int(100.0 * block_count * block_size / total_size), 100)
    bar = '[{}>{}]'.format('='*(percentage//4), ' '*(25-percentage//4))
    sys.stdout.write('{} {:3d}%\r'.format(bar, percentage))
    sys.stdout.flush()


def download_file(baseurl, filename, working_dir):
    fullpath = os.path.join(working_dir, filename)
    target_url = os.path.join(baseurl, filename)
    print('Downloading: {}'.format(target_url))
    try:
        urllib.request.urlretrieve(
            url=target_url,
            filename=fullpath,
            reporthook=progress)
        print('')
    except (OSError, urllib.error.HTTPError) as err:
        print('ERROR :{}'.format(err.code))
        print(err.reason)


def decompress_file(working_dir, filename):
    with zipfile.ZipFile(os.path.join(working_dir, filename), 'r') as z:
        z.extractall(os.path.join(working_dir, ''))


if __name__ == '__main__':
    root_dir = './../datasets/animeface'
    tmp_dir = 'tmp'
    raw_path = 'data'
    os.makedirs(root_dir, exist_ok=True)
    os.makedirs(os.path.join(root_dir, raw_path), exist_ok=True)

    baseurl = 'http://www.nurs.or.jp/~nagadomi/animeface-character-dataset/data'
    compressd_file = 'animeface-character-dataset.zip'
    filename = 'animeface-character-dataset'
    download_file(baseurl, compressd_file, root_dir)
    decompress_file(root_dir, compressd_file)
    subset(os.path.join(root_dir, filename, 'thumb'), os.path.join(root_dir, raw_path))
    shutil.rmtree(os.path.join(root_dir, filename))
