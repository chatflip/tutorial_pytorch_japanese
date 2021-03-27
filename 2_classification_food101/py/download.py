# -*- coding: utf-8 -*-
import os
import sys
import tarfile
import urllib.request


def progress(block_count, block_size, total_size):
    percentage = min(int(100.0 * block_count * block_size / total_size), 100)
    bar = "[{}>{}]".format("=" * (percentage // 4), " " * (25 - percentage // 4))
    sys.stdout.write("{} {:3d}%\r".format(bar, percentage))
    sys.stdout.flush()


def download_file(baseurl, filename, working_dir):
    fullpath = os.path.join(working_dir, filename)
    target_url = os.path.join(baseurl, filename)
    print("Downloading: {}".format(target_url))
    try:
        urllib.request.urlretrieve(
            url=target_url, filename=fullpath, reporthook=progress
        )
        print("")
    except (OSError, urllib.error.HTTPError) as err:
        print("ERROR :{}".format(err.code))
        print(err.reason)


def decompress_file(working_dir, filename):
    with tarfile.open(os.path.join(working_dir, filename), "r:gz") as tr:
        tr.extractall(path=os.path.join(working_dir, ""))


if __name__ == "__main__":
    root_dir = "./../datasets"
    baseurl = "http://data.vision.ee.ethz.ch/cvl"
    compressd_file = "food-101.tar.gz"
    filename = "food-101"
    download_file(baseurl, compressd_file, root_dir)
    decompress_file(root_dir, compressd_file)
    os.remove(os.path.join(root_dir, compressd_file))
