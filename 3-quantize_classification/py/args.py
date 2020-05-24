import argparse


def opt():
    parser = argparse.ArgumentParser(description='PyTorch AnimeFace')
    parser.add_argument('--path2db', type=str, default='data',
                        help='path to database')

    # Train Validate settings
    parser.add_argument('--batch-size', type=int, default=128,
                        help='mini-batch size in train')
    parser.add_argument('--val-batch-size', type=int, default=256,
                        help='mini-batch size in validate')
    parser.add_argument('--epochs', type=int, default=30,
                        help='number of total epochs to run')
    parser.add_argument('--num_classes', type=int, default=176,
                        help='num of classes')

    # network parameters
    parser.add_argument('--lr', type=float, default=0.0045,
                        help='initial learning rate')
    parser.add_argument('--lr-step-size', default=1, type=int,
                        help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.98, type=float,
                        help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--weight-decay', default=1e-5, type=float,
                        help='weight decay (default: 1e-5)')

    # etc
    parser.add_argument('--evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--img_size', type=int, default=256,
                        help='image size')
    parser.add_argument('--crop_size', type=int, default=224,
                        help='crop size')
    parser.add_argument('--workers', type=int, default=16,
                        help='number of data loading workers')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed for initializing training. ')
    parser.add_argument('--print-freq', type=int, default=100,
                        help='print frequency (default: 10)')

    # Mixed precision training parameters
    parser.add_argument('--apex', action='store_true',
                        help='Use apex for mixed precision training')
    parser.add_argument('--apex-opt-level', default='O1', type=str,
                        help='For apex mixed precision training'
                             'O0 for FP32 training, O1 for mixed precision training.'
                             'For further detail, see https://github.com/NVIDIA/apex/tree/master/examples/imagenet'
                        )

    # quantization settings
    parser.add_argument('--backend', default='qnnpack',
                        help='fbgemm or qnnpack')
    parser.add_argument('--num-observer-update-epochs',
                        default=5, type=int, metavar='N',
                        help='number of total epochs to update observers')
    parser.add_argument('--num-batch-norm-update-epochs', default=4,
                        type=int, metavar='N',
                        help='number of total epochs to update batch norm stats')
    parser.add_argument('--num-calibration-batches',
                        default=32, type=int, metavar='N',
                        help='number of batches of training set for \
                              observer calibration ')

    args = parser.parse_args()
    return args
