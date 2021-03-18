# -*- coding: utf-8 -*-
import os
import time

import albumentations as A
from albumentations.pytorch import ToTensorV2
import hydra
import torch
import segmentation_models_pytorch as smp


from datasets import VOCSegmentation2012
from utils import get_worker_init, seed_everything
from MlflowWriter import MlflowWriter


def load_data(args):
    cwd = hydra.utils.get_original_cwd()
    normalize = A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        max_pixel_value=255.0)

    # 画像開いたところからtensorでNNに使えるようにするまでの変形
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
        A.PadIfNeeded(min_height=args.image_height, min_width=args.image_width, always_apply=True, border_mode=0),
        A.RandomCrop(height=args.image_height, width=args.image_width, always_apply=True),
        A.IAAAdditiveGaussianNoise(p=0.2),
        A.IAAPerspective(p=0.5),
        A.OneOf([
            A.CLAHE(p=1),
            A.RandomBrightnessContrast(contrast_limit=0.0, p=1),
            A.RandomGamma(p=1),
        ], p=0.9),
        A.OneOf([
            A.IAASharpen(p=1),
            A.Blur(blur_limit=3, p=1),
            A.MotionBlur(blur_limit=3, p=1),
        ], p=0.9),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.0, p=1),
            A.HueSaturationValue(p=1),
        ], p=0.9),
        normalize,
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(args.image_height, args.image_width),
        normalize,
        ToTensorV2(),
    ])

    # VOCSegmentation2012の学習用データ設定
    train_dataset = VOCSegmentation2012(
        os.path.join(cwd, args.path2db),
        'train',
        args.num_classes,
        transform=train_transform,
    )

    # VOCSegmentation2012の評価用データ設定
    val_dataset = VOCSegmentation2012(
        os.path.join(cwd, args.path2db),
        'val',
        args.num_classes,
        transform=val_transform
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=args.backbone.batch_size,
        shuffle=True, num_workers=args.workers,
        pin_memory=True, drop_last=True,
        worker_init_fn=get_worker_init(args.seed))

    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset, batch_size=args.backbone.batch_size,
        shuffle=False, num_workers=args.workers,
        pin_memory=True, drop_last=False,
        worker_init_fn=get_worker_init(args.seed))
    return train_loader, val_loader


def write_learning_log(writer, logs, epoch, phase):
    for key in logs.keys():
        writer.log_metric(f"{phase}_{key}", logs[key], epoch)
    return writer


@hydra.main(config_path="./../config", config_name="config")
def main(args):
    print(args)
    cwd = hydra.utils.get_original_cwd()
    seed_everything(args.seed)  # 乱数テーブル固定
    os.makedirs(os.path.join(cwd, args.path2weight), exist_ok=True)
    writer = MlflowWriter(args.exp_name)
    writer.log_params_from_omegaconf_dict(args)
    torch.backends.cudnn.benchmark = True  # 再現性を無くして高速化
    train_loader, val_loader = load_data(args)

    model = getattr(smp, args.arch)(
        encoder_name=args.backbone.name,
        encoder_weights="imagenet",
        classes=args.num_classes,
        activation="softmax2d",
    )

    loss = smp.utils.losses.DiceLoss()
    metrics = [
        smp.utils.metrics.IoU(threshold=args.iou_threshold),
    ]

    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=args.lr),
    ])

    train_epoch = smp.utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device="cuda" if torch.cuda.is_available() else "cpu",
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device="cuda" if torch.cuda.is_available() else "cpu",
        verbose=True,
    )
    starttime = time.time()  # 実行時間計測(実時間)

    max_score = 0
    # 学習と評価
    for epoch in range(1, args.epochs + 1):
        print('\nEpoch: {}'.format(epoch))
        train_logs = train_epoch.run(train_loader)
        writer = write_learning_log(writer, train_logs, epoch, "train")
        valid_logs = valid_epoch.run(val_loader)
        writer = write_learning_log(writer, valid_logs, epoch, "val")

        # do something (save model, change lr, etc.)
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            weight_path = "{}/{}/{}_{}_{}_{}_{}.pth".format(
                cwd,
                args.path2weight,
                args.exp_name,
                args.arch,
                args.backbone.name,
                args.image_height,
                args.image_width,
            )
            torch.save(model, weight_path)
            print('Model saved!')

        if epoch == 25:
            optimizer.param_groups[0]['lr'] = args.lr / 10
            print('Decrease decoder learning rate to 1e-5!')

    best_model = torch.load(weight_path)
    test_epoch = smp.utils.train.ValidEpoch(
        best_model,
        loss=loss,
        metrics=metrics,
        device="cuda" if torch.cuda.is_available() else "cpu",
        verbose=False,
    )
    logs = test_epoch.run(val_loader)
    print(f"best epoch iou_score: {logs['iou_score']}")

    # Hydraの成果物をArtifactに保存
    writer.log_artifact(weight_path)
    writer.log_artifact(os.path.join(os.getcwd(), '.hydra/config.yaml'))
    writer.log_artifact(os.path.join(os.getcwd(), '.hydra/hydra.yaml'))
    writer.log_artifact(os.path.join(os.getcwd(), '.hydra/overrides.yaml'))
    writer.log_artifact(os.path.join(os.getcwd(), 'main.log'))
    writer.set_terminated()  # mlflow用のwriter閉じる

    # 実行時間表示
    endtime = time.time()
    interval = endtime - starttime
    print('elapsed time = {0:d}h {1:d}m {2:d}s'.format(
        int(interval / 3600),
        int((interval % 3600) / 60),
        int((interval % 3600) % 60)))


if __name__ == '__main__':
    main()
