import argparse


def opt():
    parser = argparse.ArgumentParser(description="PyTorch AnimeFace")
    parser.add_argument("--path2db", type=str, default="data/",
                        help="path to database")
    parser.add_argument("--path2weight", type=str, default="weight/",
                        help="path to weight")
    # Train Test settings
    parser.add_argument("--batch-size", type=int, default=32, metavar="N",
                        help="input batch size for training (default: 50)")
    parser.add_argument("--test-batch-size", type=int, default=64, metavar="N",
                        help="input batch size for testing (default: 1000)")
    parser.add_argument("--epochs", type=int, default=50, metavar="N",
                        help="number of epochs to train (default: 10)")
    parser.add_argument("--save-interval", type=int, default=10,
                        help="save every N epoch")
    parser.add_argument("--numof_classes", type=int, default=176,
                        help="num of classes")
    # network parameters
    parser.add_argument("--lr", type=float, default= 0.001, metavar="LR",
                        help="learning rate (default: 0.001)")
    parser.add_argument("--momentum", type=float, default=0.9, metavar="M",
                        help="SGD momentum (default: 0.9)")
    parser.add_argument("--weight_decay", default=0.0005, type=float,
                        help="weight decay")
    # etc
    parser.add_argument("--no-cuda", action="store_true", default=False,
                        help="disables CUDA training")
    parser.add_argument("--img_size", type=int, default=300,
                        help="image size")
    parser.add_argument("--crop_size", type=int, default=500,
                        help="crop size")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="num of pallarel threads(dataloader)")
    parser.add_argument("--seed", type=int, default=1, metavar="S",
                        help="random seed (default: 1)")
    parser.add_argument("--log-interval", type=int, default=100, metavar="N",
                        help="how many batches to wait"
                             "before logging training status")
    args = parser.parse_args()
    return args