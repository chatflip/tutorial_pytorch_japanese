import argparse


def opt():
    parser = argparse.ArgumentParser(description='PyTorch AnimeFace')
    parser.add_argument('--path2db', type=str, default='data',
                        help='path to database')

    # Train Validate settings
    parser.add_argument('--batch-size', type=int, default=256,
                        help='mini-batch size in train')
    parser.add_argument('--val-batch-size', type=int, default=512,
                        help='mini-batch size in validate')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of total epochs to run')
    parser.add_argument('--num_classes', type=int, default=176,
                        help='num of classes')

    # network parameters
    parser.add_argument('--lr', type=float, default=0.01,
                        help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-04,
                        help='weight_decay')
    # etc
    parser.add_argument('--evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--resume', type=str, default='weight/AnimeFace_resnet18_best.pth',
                        help='load weight')
    parser.add_argument('--img_size', type=int, default=256,
                        help='image size')
    parser.add_argument('--crop_size', type=int, default=224,
                        help='crop size')
    parser.add_argument('--workers', type=int, default=16,
                        help='number of data loading workers')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed for initializing training. ')
    parser.add_argument('--print-freq', type=int, default=10,
                        help='print frequency (default: 10)')
    parser.add_argument('--use_multi_gpu', type=bool, default=False,
                        help='use multi gpu')
    args = parser.parse_args()
    return args
