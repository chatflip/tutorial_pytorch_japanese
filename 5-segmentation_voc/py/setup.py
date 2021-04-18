import os
import sys
import tarfile
import urllib.request


def progress(block_count, block_size, total_size):
    percentage = min(int(100.0 * block_count * block_size / total_size), 100)
    bar = "[{}>{}]".format("=" * (percentage // 4), " " * (25 - percentage // 4))
    sys.stdout.write("{} {:3d}%\r".format(bar, percentage))
    sys.stdout.flush()


def download_file(baseurl, filename, raw_path):
    if os.path.exists(os.path.join(raw_path, filename)):
        print("File exists: {}".format(filename))
    else:
        print("Downloading: {}".format(filename))
        try:
            urllib.request.urlretrieve(
                url=baseurl + filename,
                filename=os.path.join(raw_path, filename),
                reporthook=progress,
            )
            print("")
        except (OSError, urllib.error.HTTPError) as err:
            print("ERROR :{}".format(err.code))
            print(err.reason)


def decompress_file(working_dir, filename):
    with tarfile.open(os.path.join(working_dir, filename), "r:*") as tr:
        tr.extractall(path=os.path.join(working_dir, ""))


if __name__ == "__main__":
    root_dir = "./../datasets"
    baseurl = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/"
    compressd_file = "VOCtrainval_11-May-2012.tar"
    os.makedirs(root_dir, exist_ok=True)
    download_file(baseurl, compressd_file, root_dir)
    decompress_file(root_dir, compressd_file)
    os.remove(os.path.join(root_dir, compressd_file))
