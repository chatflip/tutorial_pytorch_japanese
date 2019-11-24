import argparse


def opt():
    parser = argparse.ArgumentParser(description='PyTorch AnimeFace')
    parser.add_argument('--path2db', type=str, default='data',
                        help='path to database')
    parser.add_argument('--path2weight', type=str, default='weight',
                        help='path to weight')

    # Train Validate settings
    parser.add_argument('--batch-size', type=int, default=64,
                        help='mini-batch size in train')
    parser.add_argument('--val-batch-size', type=int, default=128,
                        help='mini-batch size in validate')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of total epochs to run')
    parser.add_argument('--save-freq', type=int, default=10,
                        help='save every N epoch')
    parser.add_argument('--numof_classes', type=int, default=176,
                        help='num of classes')

    # network parameters
    parser.add_argument('--lr', type=float, default=0.001,
                        help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # etc
    parser.add_argument('--evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--img_size', type=int, default=256,
                        help='image size')
    parser.add_argument('--crop_size', type=int, default=224,
                        help='crop size')
    parser.add_argument('--workers', type=int, default=4,
                        help='number of data loading workers')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed for initializing training. ')
    parser.add_argument('--print-freq', type=int, default=100,
                        help='print frequency (default: 10)')

    args = parser.parse_args()
    return args
