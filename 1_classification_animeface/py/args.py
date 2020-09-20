import argparse


def opt():
    parser = argparse.ArgumentParser(description='PyTorch AnimeFace')
    parser.add_argument('--exp_name', type=str, default='animeface',
                        help='prefix')
    parser.add_argument('--path2db', type=str, default='./../datasets/animeface/data',
                        help='path to database')
    parser.add_argument('--path2weight', type=str, default='weight',
                        help='path to database')

    # network settings
    parser.add_argument('--num_classes', type=int, default=176,
                        help='num of classes')

    # training settings
    parser.add_argument('--lr', type=float, default=0.0045,
                        help='initial learning rate')
    parser.add_argument('--lr-step-size', default=1, type=int,
                        help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.98, type=float,
                        help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='weight_decay')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='mini-batch size in train')
    parser.add_argument('--start_epoch', type=int, default=1,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of total epochs to run')

    # etc
    parser.add_argument('--evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--restart', action='store_true',
                        help='restart training')
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

    # Mixed precision training parameters
    parser.add_argument('--apex', action='store_true',
                        help='Use apex for mixed precision training')
    args = parser.parse_args()
    return args
