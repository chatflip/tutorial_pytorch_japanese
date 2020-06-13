import os

import torch
import torch.nn as nn

import utils
from engine import train_one_epoch, evaluate
from transform import get_transform
from dataset import PennFudanDataset
from model import get_instance_segmentation_model


def main():
    # フォルダが存在してなければ作る
    if not os.path.exists('weight'):
        os.mkdir('weight')

    # use our dataset and defined transformations
    dataset = PennFudanDataset('data/PennFudanPed', get_transform(train=True))
    dataset_test = PennFudanDataset('data/PennFudanPed', get_transform(train=False))

    # split the dataset in train and test set
    torch.manual_seed(1)

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=2, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    multigpu = torch.cuda.device_count() > 1  # グラボ2つ以上ならmultigpuにする

    # our dataset has two classes only - background and person
    num_classes = 2

    # get the model using our helper function
    model = get_instance_segmentation_model(num_classes)
    # move model to the right device
    model.to(device)


    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

    # and a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)
    model_without_ddp = model
    if multigpu:
        model = nn.DataParallel(model)
        model_without_ddp = model.module

    # let's train it for 10 epochs
    num_epochs = 10

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

        saved_weight = 'weight/Penn-Fudan_maskrcnn_epoch{}.pth'.format(epoch)
        torch.save(model_without_ddp.cpu().state_dict(), saved_weight)
        model.to(device)


if __name__ == '__main__':
    main()