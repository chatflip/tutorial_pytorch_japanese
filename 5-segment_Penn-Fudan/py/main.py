import os

import torch
import torch.nn as nn

import utils
from args import opt
from engine import train_one_epoch, evaluate
from transform import get_transform
from dataset import PennFudanDataset
from model import get_instance_segmentation_model

try:
    from apex import amp
except ImportError:
    amp = None


def load_data(args):
    # use our dataset and defined transformations
    train_dataset = PennFudanDataset(
        os.path.join(args.path2db, 'PennFudanPed'), get_transform(train=True))
    val_dataset = PennFudanDataset(
        os.path.join(args.path2db, 'PennFudanPed'), get_transform(train=False))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        val_sampler = torch.utils.data.SequentialSampler(val_dataset)
    return train_dataset, val_dataset, train_sampler, val_sampler


def main():
    args = opt()
    print(args)
    utils.seed_everything(args.seed)  # 乱数テーブル固定
    collate_fn = utils.get_worker_init()
    if args.apex and amp is None:
        raise RuntimeError("Failed to import apex. Please install apex from https://www.github.com/nvidia/apex "
                           "to enable mixed-precision training.")
    # フォルダが存在してなければ作る
    if not os.path.exists('weight'):
        os.mkdir('weight')

    utils.init_distributed_mode(args)
    ngpus_per_node = torch.cuda.device_count()

    if args.distributed:
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)

    train_dataset, val_dataset, train_sampler, val_sampler = load_data(args)

    # define training and validation data loaders
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=args.batch_size,
        sampler = train_sampler, num_workers=args.workers,
        pin_memory=True, drop_last=True,
        collate_fn=collate_fn)

    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset, batch_size=args.batch_size,
        sampler=val_sampler, num_workers=args.workers,
        pin_memory=True, drop_last=False,
        collate_fn=collate_fn)


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # get the model using our helper function
    model = get_instance_segmentation_model(args.num_classes)
    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)


    # --evaluate
    # --resume

    if args.apex:
        model, optimizer = amp.initialize(
            model, optimizer,
            opt_level=args.apex_opt_level
        )

    model_without_ddp = model
    if args.distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)

    for epoch in range(args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=args.print_freq)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, val_loader, device=device)

        saved_weight = 'weight/Penn-Fudan_maskrcnn_epoch{}.pth'.format(epoch)
        torch.save(model_without_ddp.cpu().state_dict(), saved_weight)
        model.to(device)


if __name__ == '__main__':
    main()
