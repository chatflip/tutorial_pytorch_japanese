# -*- coding: utf-8 -*-
from __future__ import print_function

from loadDB import VOCdetectionDB


if __name__ == "__main__":
    db = VOCdetectionDB(root='data/VOCdevkit/VOC2012', phase='val')
