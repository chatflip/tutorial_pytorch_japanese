(pt16) pcd@pcd:/mnt/hdd/github/tutorial_pytorch_japanese/2_classification_food101_ddp$ bash train.sh Namespace(apex=True, apex_opt_level='O1', batch_size=256, crop_size=224, dist_backend='nccl', dist_url='env://', epochs=20, evaluate=False, exp_name='food101', gpu=None, img_size=256, lr=0.0045, lr_gamma=0.98, lr_step_size=1, momentum=0.9, multiprocessing_distributed=False, num_classes=101, path2db='./../datasets/food-101', path2weight='weight', print_freq=100, rank=-1, restart=False, seed=1, start_epoch=1, sync_bn=False, weight_decay=4e-05, workers=16, world_size=1)
Namespace(apex=True, apex_opt_level='O1', batch_size=256, crop_size=224, dist_backend='nccl', dist_url='env://', epochs=20, evaluate=False, exp_name='food101', gpu=None, img_size=256, lr=0.0045, lr_gamma=0.98, lr_step_size=1, momentum=0.9, multiprocessing_distributed=False, num_classes=101, path2db='./../datasets/food-101', path2weight='weight', print_freq=100, rank=-1, restart=False, seed=1, start_epoch=1, sync_bn=False, weight_decay=4e-05, workers=16, world_size=1)
| distributed init (rank 0): env://
| distributed init (rank 1): env://
Selected optimization level O1:  Insert automatic casts around Pytorch functions and Tensor methods.

Defaults for this optimization level are:
enabled                : True
opt_level              : O1
cast_model_type        : None
patch_torch_functions  : True
keep_batchnorm_fp32    : None
master_weights         : None
loss_scale             : dynamic
Processing user overrides (additional kwargs that are not None)...
After processing overrides, optimization options are:
enabled                : True
opt_level              : O1
cast_model_type        : None
patch_torch_functions  : True
keep_batchnorm_fp32    : None
master_weights         : None
loss_scale             : dynamic
Epoch: [1] [  0/295] eta: 0:34:17 loss: 4.6944 (4.6944) acc1: 0.0000 (0.0000) acc5: 3.9062 (3.9062) time: 6.9739 data: 1.7344
Epoch: [1] [100/295] eta: 0:00:53 loss: 3.0450 (3.8585) acc1: 32.8125 (19.3920) acc5: 61.7188 (39.7200) time: 0.2008 data: 0.0003
Epoch: [1] [200/295] eta: 0:00:22 loss: 2.3609 (3.2121) acc1: 44.5312 (30.8147) acc5: 72.6562 (54.7147) time: 0.2032 data: 0.0002
Epoch: [1] Total time: 0:01:06
Validate:  [ 0/99]  eta: 0:03:15  loss: 1.5792 (1.5792)  acc1: 59.3750 (59.3750)  acc5: 86.7188 (86.7188)  time: 1.9725  data: 1.9204
Validate: Total time: 0:00:19
Acc@1 best:  62.88%
Epoch: [2] [  0/295] eta: 0:08:47 loss: 2.0886 (2.0886) acc1: 49.2188 (49.2188) acc5: 77.3438 (77.3438) time: 1.7894 data: 1.5167
Epoch: [2] [100/295] eta: 0:00:42 loss: 1.9604 (1.9501) acc1: 53.1250 (51.6631) acc5: 78.9062 (78.2178) time: 0.2003 data: 0.0003
Epoch: [2] [200/295] eta: 0:00:19 loss: 1.8605 (1.9169) acc1: 53.9062 (52.3204) acc5: 79.6875 (78.6419) time: 0.2000 data: 0.0002
Epoch: [2] Total time: 0:01:00
Validate:  [ 0/99]  eta: 0:03:04  loss: 1.3823 (1.3823)  acc1: 64.8438 (64.8438)  acc5: 87.5000 (87.5000)  time: 1.8671  data: 1.8148
Validate: Total time: 0:00:18
Acc@1 best:  68.06%
Epoch: [3] [  0/295] eta: 0:10:52 loss: 1.8684 (1.8684) acc1: 54.6875 (54.6875) acc5: 75.0000 (75.0000) time: 2.2122 data: 1.9927
Epoch: [3] [100/295] eta: 0:00:42 loss: 1.6695 (1.7235) acc1: 58.5938 (56.7992) acc5: 80.4688 (81.3660) time: 0.1952 data: 0.0002
Epoch: [3] [200/295] eta: 0:00:19 loss: 1.6803 (1.6953) acc1: 58.5938 (57.5832) acc5: 82.8125 (81.7980) time: 0.1929 data: 0.0002
Epoch: [3] Total time: 0:01:00
Validate:  [ 0/99]  eta: 0:02:52  loss: 1.0677 (1.0677)  acc1: 68.7500 (68.7500)  acc5: 93.7500 (93.7500)  time: 1.7415  data: 1.6735
Validate: Total time: 0:00:18
Acc@1 best:  70.69%
Epoch: [4] [  0/295] eta: 0:09:25 loss: 1.6425 (1.6425) acc1: 59.3750 (59.3750) acc5: 82.0312 (82.0312) time: 1.9186 data: 1.6888
Epoch: [4] [100/295] eta: 0:00:42 loss: 1.6473 (1.5785) acc1: 59.3750 (60.1795) acc5: 82.0312 (83.3772) time: 0.1987 data: 0.0003
Epoch: [4] [200/295] eta: 0:00:19 loss: 1.5306 (1.5681) acc1: 60.9375 (60.2534) acc5: 85.1562 (83.4849) time: 0.1966 data: 0.0003
Epoch: [4] Total time: 0:01:00
Validate:  [ 0/99]  eta: 0:02:58  loss: 1.0793 (1.0793)  acc1: 71.0938 (71.0938)  acc5: 90.6250 (90.6250)  time: 1.8008  data: 1.7336
Validate: Total time: 0:00:18
Acc@1 best:  73.45%
Epoch: [5] [  0/295] eta: 0:09:05 loss: 1.6063 (1.6063) acc1: 57.0312 (57.0312) acc5: 83.5938 (83.5938) time: 1.8508 data: 1.6166
Epoch: [5] [100/295] eta: 0:00:42 loss: 1.4592 (1.4658) acc1: 63.2812 (62.8249) acc5: 85.9375 (84.9242) time: 0.1982 data: 0.0003
Epoch: [5] [200/295] eta: 0:00:19 loss: 1.4234 (1.4650) acc1: 63.2812 (62.7371) acc5: 85.9375 (84.9580) time: 0.1972 data: 0.0002
Epoch: [5] Total time: 0:01:00
Validate:  [ 0/99]  eta: 0:03:00  loss: 1.1504 (1.1504)  acc1: 67.1875 (67.1875)  acc5: 89.8438 (89.8438)  time: 1.8265  data: 1.7725
Validate: Total time: 0:00:18
Acc@1 best:  73.70%
Epoch: [6] [  0/295] eta: 0:08:36 loss: 1.3168 (1.3168) acc1: 67.9688 (67.9688) acc5: 89.0625 (89.0625) time: 1.7493 data: 1.5243
Epoch: [6] [100/295] eta: 0:00:42 loss: 1.3908 (1.4076) acc1: 62.5000 (63.8691) acc5: 85.1562 (85.9839) time: 0.1971 data: 0.0003
Epoch: [6] [200/295] eta: 0:00:19 loss: 1.3284 (1.4089) acc1: 64.8438 (63.8176) acc5: 86.7188 (85.7354) time: 0.1984 data: 0.0003
Epoch: [6] Total time: 0:01:00
Validate:  [ 0/99]  eta: 0:03:07  loss: 0.9749 (0.9749)  acc1: 68.7500 (68.7500)  acc5: 92.1875 (92.1875)  time: 1.8940  data: 1.8414
Validate: Total time: 0:00:18
Acc@1 best:  75.51%
Epoch: [7] [  0/295] eta: 0:08:13 loss: 1.2766 (1.2766) acc1: 72.6562 (72.6562) acc5: 89.0625 (89.0625) time: 1.6717 data: 1.4632
Epoch: [7] [100/295] eta: 0:00:42 loss: 1.3688 (1.3516) acc1: 63.2812 (65.1609) acc5: 84.3750 (86.4325) time: 0.1969 data: 0.0003
Epoch: [7] [200/295] eta: 0:00:19 loss: 1.3430 (1.3508) acc1: 64.0625 (65.3451) acc5: 87.5000 (86.4933) time: 0.1959 data: 0.0002
Epoch: [7] Total time: 0:00:59
Validate:  [ 0/99]  eta: 0:03:19  loss: 0.9281 (0.9281)  acc1: 75.7812 (75.7812)  acc5: 93.7500 (93.7500)  time: 2.0178  data: 1.9516
Validate: Total time: 0:00:18
Acc@1 best:  76.08%
Epoch: [8] [  0/295] eta: 0:08:58 loss: 1.1494 (1.1494) acc1: 71.8750 (71.8750) acc5: 91.4062 (91.4062) time: 1.8271 data: 1.6007
Epoch: [8] [100/295] eta: 0:00:41 loss: 1.3066 (1.3116) acc1: 65.6250 (66.4217) acc5: 87.5000 (87.2525) time: 0.1958 data: 0.0003
Epoch: [8] [200/295] eta: 0:00:19 loss: 1.3039 (1.3060) acc1: 64.8438 (66.5034) acc5: 87.5000 (87.2746) time: 0.1952 data: 0.0002
Epoch: [8] Total time: 0:00:59
Validate:  [ 0/99]  eta: 0:02:43  loss: 0.8833 (0.8833)  acc1: 73.4375 (73.4375)  acc5: 93.7500 (93.7500)  time: 1.6523  data: 1.5984
Validate: Total time: 0:00:18
Acc@1 best:  77.32%
Epoch: [9] [  0/295] eta: 0:08:17 loss: 1.0034 (1.0034) acc1: 74.2188 (74.2188) acc5: 92.1875 (92.1875) time: 1.6857 data: 1.4694
Epoch: [9] [100/295] eta: 0:00:42 loss: 1.2927 (1.2713) acc1: 66.4062 (66.9322) acc5: 87.5000 (87.4768) time: 0.1992 data: 0.0003
Epoch: [9] [200/295] eta: 0:00:19 loss: 1.2787 (1.2671) acc1: 66.4062 (67.3080) acc5: 87.5000 (87.5155) time: 0.1933 data: 0.0002
Epoch: [9] Total time: 0:00:59
Validate:  [ 0/99]  eta: 0:02:42  loss: 0.8635 (0.8635)  acc1: 72.6562 (72.6562)  acc5: 94.5312 (94.5312)  time: 1.6455  data: 1.5798
Validate: Total time: 0:00:18
Acc@1 best:  77.94%
Epoch: [10] [  0/295] eta: 0:09:30 loss: 1.1422 (1.1422) acc1: 70.3125 (70.3125) acc5: 88.2812 (88.2812) time: 1.9324 data: 1.7186
Epoch: [10] [100/295] eta: 0:00:42 loss: 1.2045 (1.2403) acc1: 67.9688 (67.6284) acc5: 87.5000 (87.5928) time: 0.1952 data: 0.0002
Epoch: [10] [200/295] eta: 0:00:19 loss: 1.2156 (1.2384) acc1: 67.9688 (67.6772) acc5: 88.2812 (87.7526) time: 0.1975 data: 0.0003
Epoch: [10] Total time: 0:01:00
Validate:  [ 0/99]  eta: 0:02:59  loss: 0.8618 (0.8618)  acc1: 77.3438 (77.3438)  acc5: 92.9688 (92.9688)  time: 1.8082  data: 1.7541
Validate: Total time: 0:00:18
Acc@1 best:  78.50%
Epoch: [11] [  0/295] eta: 0:08:18 loss: 1.0442 (1.0442) acc1: 74.2188 (74.2188) acc5: 88.2812 (88.2812) time: 1.6909 data: 1.4697
Epoch: [11] [100/295] eta: 0:00:42 loss: 1.2114 (1.2220) acc1: 69.5312 (68.2936) acc5: 88.2812 (87.8868) time: 0.1986 data: 0.0003
Epoch: [11] [200/295] eta: 0:00:19 loss: 1.2219 (1.2091) acc1: 67.9688 (68.5440) acc5: 89.0625 (88.1025) time: 0.1971 data: 0.0003
Epoch: [11] Total time: 0:01:00
Validate:  [ 0/99]  eta: 0:03:01  loss: 0.7344 (0.7344)  acc1: 75.0000 (75.0000)  acc5: 95.3125 (95.3125)  time: 1.8324  data: 1.7639
Validate: Total time: 0:00:18
Acc@1 best:  78.79%
Epoch: [12] [  0/295] eta: 0:10:24 loss: 1.5340 (1.5340) acc1: 59.3750 (59.3750) acc5: 87.5000 (87.5000) time: 2.1178 data: 1.8965
Epoch: [12] [100/295] eta: 0:00:42 loss: 1.1808 (1.1778) acc1: 67.9688 (69.3611) acc5: 89.0625 (88.7299) time: 0.1972 data: 0.0002
Epoch: [12] [200/295] eta: 0:00:19 loss: 1.1566 (1.1738) acc1: 67.9688 (69.4691) acc5: 89.8438 (88.9187) time: 0.1980 data: 0.0003
Epoch: [12] Total time: 0:01:00
Validate:  [ 0/99]  eta: 0:03:00  loss: 0.7258 (0.7258)  acc1: 79.6875 (79.6875)  acc5: 95.3125 (95.3125)  time: 1.8243  data: 1.7702
Validate: Total time: 0:00:18
Acc@1 best:  78.96%
Epoch: [13] [  0/295] eta: 0:08:53 loss: 1.2968 (1.2968) acc1: 64.8438 (64.8438) acc5: 89.0625 (89.0625) time: 1.8073 data: 1.5979
Epoch: [13] [100/295] eta: 0:00:41 loss: 1.1172 (1.1200) acc1: 69.5312 (70.8230) acc5: 89.0625 (89.7509) time: 0.1979 data: 0.0003
Epoch: [13] [200/295] eta: 0:00:19 loss: 1.1222 (1.1358) acc1: 71.0938 (70.4213) acc5: 89.0625 (89.4317) time: 0.1985 data: 0.0003
Epoch: [13] Total time: 0:00:59
Validate:  [ 0/99]  eta: 0:03:12  loss: 0.8282 (0.8282)  acc1: 77.3438 (77.3438)  acc5: 92.9688 (92.9688)  time: 1.9486  data: 1.8914
Validate: Total time: 0:00:18
Acc@1 best:  79.80%
Epoch: [14] [  0/295] eta: 0:08:14 loss: 1.0136 (1.0136) acc1: 73.4375 (73.4375) acc5: 89.0625 (89.0625) time: 1.6766 data: 1.4712
Epoch: [14] [100/295] eta: 0:00:42 loss: 1.0618 (1.1015) acc1: 71.8750 (71.2407) acc5: 89.8438 (89.7432) time: 0.1952 data: 0.0002
Epoch: [14] [200/295] eta: 0:00:19 loss: 1.1058 (1.1165) acc1: 71.0938 (70.7906) acc5: 89.0625 (89.6416) time: 0.1965 data: 0.0003
Gradient overflow.  Skipping step, loss scaler 0 reducing loss scale to 131072.0
Epoch: [14] Total time: 0:01:00
Validate:  [ 0/99]  eta: 0:03:09  loss: 0.7953 (0.7953)  acc1: 78.1250 (78.1250)  acc5: 96.0938 (96.0938)  time: 1.9140  data: 1.8503
Validate: Total time: 0:00:18
Acc@1 :  79.67%
Epoch: [15] [  0/295] eta: 0:08:40 loss: 1.0726 (1.0726) acc1: 73.4375 (73.4375) acc5: 89.0625 (89.0625) time: 1.7654 data: 1.5408
Epoch: [15] [100/295] eta: 0:00:42 loss: 1.0492 (1.0794) acc1: 70.3125 (71.3645) acc5: 89.8438 (90.0603) time: 0.1963 data: 0.0003
Epoch: [15] [200/295] eta: 0:00:19 loss: 1.0730 (1.0860) acc1: 71.0938 (71.2414) acc5: 89.0625 (90.0303) time: 0.1950 data: 0.0003
Epoch: [15] Total time: 0:00:59
Validate:  [ 0/99]  eta: 0:03:05  loss: 0.7502 (0.7502)  acc1: 77.3438 (77.3438)  acc5: 96.0938 (96.0938)  time: 1.8696  data: 1.8146
Validate: Total time: 0:00:18
Acc@1 best:  80.11%
Epoch: [16] [  0/295] eta: 0:09:58 loss: 1.0075 (1.0075) acc1: 75.0000 (75.0000) acc5: 92.1875 (92.1875) time: 2.0274 data: 1.7801
Gradient overflow.  Skipping step, loss scaler 0 reducing loss scale to 65536.0
Epoch: [16] [100/295] eta: 0:00:42 loss: 1.0773 (1.0809) acc1: 71.8750 (71.9291) acc5: 90.6250 (90.1686) time: 0.1980 data: 0.0003
Epoch: [16] [200/295] eta: 0:00:19 loss: 1.0849 (1.0738) acc1: 71.0938 (71.9722) acc5: 89.8438 (90.0614) time: 0.1985 data: 0.0003
Epoch: [16] Total time: 0:01:00
Validate:  [ 0/99]  eta: 0:03:20  loss: 0.7885 (0.7885)  acc1: 72.6562 (72.6562)  acc5: 93.7500 (93.7500)  time: 2.0217  data: 1.9693
Validate: Total time: 0:00:18
Acc@1 best:  80.31%
Epoch: [17] [  0/295] eta: 0:09:03 loss: 1.0475 (1.0475) acc1: 75.0000 (75.0000) acc5: 90.6250 (90.6250) time: 1.8413 data: 1.6331
Epoch: [17] [100/295] eta: 0:00:41 loss: 1.0209 (1.0471) acc1: 72.6562 (72.5712) acc5: 89.8438 (90.5476) time: 0.1967 data: 0.0002
Epoch: [17] [200/295] eta: 0:00:19 loss: 1.1007 (1.0559) acc1: 71.8750 (72.4425) acc5: 89.0625 (90.4423) time: 0.1967 data: 0.0002
Epoch: [17] Total time: 0:00:59
Validate:  [ 0/99]  eta: 0:02:54  loss: 0.7643 (0.7643)  acc1: 74.2188 (74.2188)  acc5: 95.3125 (95.3125)  time: 1.7590  data: 1.6926
Validate: Total time: 0:00:18
Acc@1 best:  80.86%
Epoch: [18] [  0/295] eta: 0:09:55 loss: 0.9826 (0.9826) acc1: 72.6562 (72.6562) acc5: 92.9688 (92.9688) time: 2.0203 data: 1.8060
Epoch: [18] [100/295] eta: 0:00:42 loss: 1.0815 (1.0485) acc1: 71.8750 (72.3700) acc5: 89.0625 (90.1222) time: 0.1953 data: 0.0002
Epoch: [18] [200/295] eta: 0:00:19 loss: 1.0230 (1.0496) acc1: 72.6562 (72.3997) acc5: 91.4062 (90.2596) time: 0.1972 data: 0.0003
Epoch: [18] Total time: 0:00:59
Validate:  [ 0/99]  eta: 0:03:12  loss: 0.7291 (0.7291)  acc1: 78.1250 (78.1250)  acc5: 95.3125 (95.3125)  time: 1.9421  data: 1.8671
Validate: Total time: 0:00:18
Acc@1 best:  81.10%
Epoch: [19] [  0/295] eta: 0:08:15 loss: 0.9726 (0.9726) acc1: 73.4375 (73.4375) acc5: 90.6250 (90.6250) time: 1.6782 data: 1.4647
Epoch: [19] [100/295] eta: 0:00:42 loss: 0.9369 (1.0026) acc1: 73.4375 (73.2519) acc5: 92.1875 (91.1278) time: 0.1984 data: 0.0003
Epoch: [19] [200/295] eta: 0:00:19 loss: 1.0717 (1.0219) acc1: 71.8750 (72.9050) acc5: 89.8438 (90.7688) time: 0.1952 data: 0.0003
Epoch: [19] Total time: 0:01:00
Validate:  [ 0/99]  eta: 0:02:45  loss: 0.6566 (0.6566)  acc1: 80.4688 (80.4688)  acc5: 94.5312 (94.5312)  time: 1.6730  data: 1.6093
Validate: Total time: 0:00:18
Acc@1 :  80.65%
Epoch: [20] [  0/295] eta: 0:08:58 loss: 1.1575 (1.1575) acc1: 70.3125 (70.3125) acc5: 89.8438 (89.8438) time: 1.8241 data: 1.6081
Epoch: [20] [100/295] eta: 0:00:41 loss: 0.9846 (0.9912) acc1: 73.4375 (73.8475) acc5: 91.4062 (91.2206) time: 0.1977 data: 0.0003
Epoch: [20] [200/295] eta: 0:00:19 loss: 1.0020 (0.9883) acc1: 72.6562 (73.8067) acc5: 91.4062 (91.2741) time: 0.1982 data: 0.0002
Epoch: [20] Total time: 0:01:00
Validate:  [ 0/99]  eta: 0:02:57  loss: 0.6850 (0.6850)  acc1: 78.9062 (78.9062)  acc5: 97.6562 (97.6562)  time: 1.7894  data: 1.7363
Validate: Total time: 0:00:18
Acc@1 :  80.94%
elapsed time = 0h 26m 21s
