import argparse


def opt():
    parser = argparse.ArgumentParser(description='PyTorch Penn-Fudan')
    parser.add_argument('--path2db', type=str, default='data',
                        help='path to database')

    # Train Validate settings
    parser.add_argument('--batch-size', type=int, default=8,
                        help='mini-batch size in train')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of total epochs to run')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='num of classes')

    # network parameters
    parser.add_argument('--lr', type=float, default=0.005,
                        help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--weight_decay', type=float, default=0.0005,
                        help='weight_decay')
    parser.add_argument('--lr_step_size', type=int, default=3,
                        help='initial learning rate')
    parser.add_argument('--lr_gamma', type=float, default=0.1,
                        help='initial learning rate')

    # etc
    parser.add_argument('--workers', type=int, default=16,
                        help='number of data loading workers')
    parser.add_argument('--print-freq', type=int, default=10,
                        help='print frequency (default: 10)')
    parser.add_argument('--evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--resume', type=str, default='weight/AnimeFace_resnet18_best.pth',
                        help='load weight')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed for initializing training. ')

    # distribution settings
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')

    args = parser.parse_args()
    return args
