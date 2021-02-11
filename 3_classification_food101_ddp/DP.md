(pt16) pcd@pcd:/mnt/hdd/github/tutorial_pytorch_japanese/3_classification_food101_dp$ bash train.sh 
Namespace(apex=True, apex_opt_level='O1', batch_size=256, crop_size=224, dist_backend='nccl', dist_url='env://', epochs=20, evaluate=False, exp_name='food101', gpu=None, img_size=256, lr=0.0045, lr_gamma=0.98, lr_step_size=1, momentum=0.9, multiprocessing_distributed=False, num_classes=101, path2db='./../datasets/food-101', path2weight='weight', print_freq=100, rank=-1, restart=False, seed=1, start_epoch=1, sync_bn=False, weight_decay=4e-05, workers=16, world_size=1)
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
Epoch: [1][  0/295] Time  7.451 ( 7.451)    Data  2.838 ( 2.838)    Loss 4.64200 (4.64200)  Acc@1   1.95 (  1.95)   Acc@5   4.69 (  4.69)
Epoch: [1][100/295] Time  0.209 ( 0.288)    Data  0.000 ( 0.028)    Loss 2.95192 (3.86248)  Acc@1  37.89 ( 19.54)   Acc@5  62.11 ( 39.94)
Epoch: [1][200/295] Time  0.221 ( 0.252)    Data  0.000 ( 0.014)    Loss 2.22808 (3.21671)  Acc@1  42.97 ( 30.75)   Acc@5  76.17 ( 54.61)
Validate: [ 0/99]   Time  4.631 ( 4.631)    Data  4.384 ( 4.384)    Loss 2.86110 (2.86110)  Acc@1  17.19 ( 17.19)   Acc@5  61.72 ( 61.72)
Validate: [99/99]   Time  0.045 ( 0.193)    Data  0.000 ( 0.082)    Loss 1.42859 (1.51111)  Acc@1  66.05 ( 62.53)   Acc@5  88.27 ( 87.37)
Acc@1 best:  62.53%
Epoch: [2][  0/295] Time  3.219 ( 3.219)    Data  2.945 ( 2.945)    Loss 2.10698 (2.10698)  Acc@1  45.70 ( 45.70)   Acc@5  75.78 ( 75.78)
Epoch: [2][100/295] Time  0.209 ( 0.262)    Data  0.000 ( 0.037)    Loss 1.82766 (1.91326)  Acc@1  55.86 ( 53.24)   Acc@5  79.30 ( 79.20)
Epoch: [2][200/295] Time  0.220 ( 0.239)    Data  0.000 ( 0.019)    Loss 1.76697 (1.84046)  Acc@1  52.73 ( 54.62)   Acc@5  81.25 ( 80.18)
Validate: [ 0/99]   Time  3.331 ( 3.331)    Data  3.200 ( 3.200)    Loss 2.45396 (2.45396)  Acc@1  29.69 ( 29.69)   Acc@5  73.83 ( 73.83)
Validate: [99/99]   Time  0.045 ( 0.191)    Data  0.000 ( 0.078)    Loss 0.79686 (1.10410)  Acc@1  79.01 ( 71.18)   Acc@5  92.59 ( 91.81)
Acc@1 best:  71.18%
Epoch: [3][  0/295] Time  3.523 ( 3.523)    Data  3.103 ( 3.103)    Loss 1.30187 (1.30187)  Acc@1  67.19 ( 67.19)   Acc@5  87.11 ( 87.11)
Epoch: [3][100/295] Time  0.222 ( 0.262)    Data  0.000 ( 0.039)    Loss 1.66785 (1.60007)  Acc@1  57.03 ( 59.92)   Acc@5  82.42 ( 83.11)
Epoch: [3][200/295] Time  0.208 ( 0.239)    Data  0.000 ( 0.020)    Loss 1.34933 (1.58461)  Acc@1  65.62 ( 60.27)   Acc@5  87.11 ( 83.45)
Validate: [ 0/99]   Time  3.254 ( 3.254)    Data  3.126 ( 3.126)    Loss 2.04099 (2.04099)  Acc@1  39.45 ( 39.45)   Acc@5  83.98 ( 83.98)
Validate: [99/99]   Time  0.045 ( 0.192)    Data  0.000 ( 0.080)    Loss 0.55434 (0.96037)  Acc@1  85.19 ( 74.17)   Acc@5  98.15 ( 93.26)
Acc@1 best:  74.17%
Epoch: [4][  0/295] Time  3.240 ( 3.240)    Data  2.955 ( 2.955)    Loss 1.41961 (1.41961)  Acc@1  61.72 ( 61.72)   Acc@5  86.72 ( 86.72)
Epoch: [4][100/295] Time  0.222 ( 0.262)    Data  0.000 ( 0.042)    Loss 1.47280 (1.44626)  Acc@1  63.28 ( 63.40)   Acc@5  86.72 ( 85.44)
Epoch: [4][200/295] Time  0.230 ( 0.239)    Data  0.000 ( 0.021)    Loss 1.34834 (1.43872)  Acc@1  62.89 ( 63.39)   Acc@5  88.67 ( 85.55)
Validate: [ 0/99]   Time  3.250 ( 3.250)    Data  3.145 ( 3.145)    Loss 2.16807 (2.16807)  Acc@1  33.98 ( 33.98)   Acc@5  80.08 ( 80.08)
Validate: [99/99]   Time  0.045 ( 0.192)    Data  0.000 ( 0.080)    Loss 0.82156 (0.91149)  Acc@1  79.01 ( 75.05)   Acc@5  95.06 ( 93.59)
Acc@1 best:  75.05%
Epoch: [5][  0/295] Time  3.957 ( 3.957)    Data  3.651 ( 3.651)    Loss 1.49635 (1.49635)  Acc@1  64.84 ( 64.84)   Acc@5  82.42 ( 82.42)
Epoch: [5][100/295] Time  0.208 ( 0.261)    Data  0.000 ( 0.037)    Loss 1.43766 (1.34882)  Acc@1  60.55 ( 65.38)   Acc@5  86.33 ( 86.53)
Epoch: [5][200/295] Time  0.225 ( 0.239)    Data  0.000 ( 0.019)    Loss 1.23077 (1.34595)  Acc@1  68.36 ( 65.37)   Acc@5  86.72 ( 86.62)
Validate: [ 0/99]   Time  3.320 ( 3.320)    Data  3.208 ( 3.208)    Loss 2.01976 (2.01976)  Acc@1  39.84 ( 39.84)   Acc@5  84.38 ( 84.38)
Validate: [99/99]   Time  0.045 ( 0.192)    Data  0.000 ( 0.078)    Loss 0.56419 (0.82428)  Acc@1  83.33 ( 77.21)   Acc@5  96.30 ( 94.50)
Acc@1 best:  77.21%
Epoch: [6][  0/295] Time  3.798 ( 3.798)    Data  3.472 ( 3.472)    Loss 1.19187 (1.19187)  Acc@1  69.14 ( 69.14)   Acc@5  88.28 ( 88.28)
Epoch: [6][100/295] Time  0.210 ( 0.259)    Data  0.000 ( 0.035)    Loss 1.11060 (1.28434)  Acc@1  71.88 ( 67.22)   Acc@5  92.58 ( 87.50)
Epoch: [6][200/295] Time  0.214 ( 0.238)    Data  0.000 ( 0.018)    Loss 0.97987 (1.28408)  Acc@1  74.22 ( 67.05)   Acc@5  91.41 ( 87.57)
Validate: [ 0/99]   Time  3.954 ( 3.954)    Data  3.782 ( 3.782)    Loss 1.75465 (1.75465)  Acc@1  47.66 ( 47.66)   Acc@5  87.50 ( 87.50)
Validate: [99/99]   Time  0.045 ( 0.191)    Data  0.000 ( 0.075)    Loss 0.50785 (0.79479)  Acc@1  88.27 ( 78.25)   Acc@5  96.30 ( 94.65)
Acc@1 best:  78.25%
Epoch: [7][  0/295] Time  3.365 ( 3.365)    Data  3.103 ( 3.103)    Loss 1.20772 (1.20772)  Acc@1  66.02 ( 66.02)   Acc@5  88.28 ( 88.28)
Epoch: [7][100/295] Time  0.225 ( 0.259)    Data  0.000 ( 0.032)    Loss 1.06261 (1.21951)  Acc@1  69.92 ( 68.50)   Acc@5  90.62 ( 88.15)
Epoch: [7][200/295] Time  0.210 ( 0.238)    Data  0.000 ( 0.016)    Loss 1.26876 (1.22129)  Acc@1  65.62 ( 68.52)   Acc@5  90.23 ( 88.30)
Validate: [ 0/99]   Time  3.255 ( 3.255)    Data  3.136 ( 3.136)    Loss 1.85207 (1.85207)  Acc@1  43.36 ( 43.36)   Acc@5  87.11 ( 87.11)
Validate: [99/99]   Time  0.045 ( 0.192)    Data  0.000 ( 0.084)    Loss 0.38770 (0.77281)  Acc@1  91.36 ( 78.57)   Acc@5  97.53 ( 94.98)
Acc@1 best:  78.57%
Epoch: [8][  0/295] Time  3.763 ( 3.763)    Data  3.472 ( 3.472)    Loss 1.05967 (1.05967)  Acc@1  69.92 ( 69.92)   Acc@5  91.41 ( 91.41)
Epoch: [8][100/295] Time  0.207 ( 0.259)    Data  0.000 ( 0.035)    Loss 1.11474 (1.17550)  Acc@1  71.48 ( 69.60)   Acc@5  87.50 ( 88.87)
Epoch: [8][200/295] Time  0.223 ( 0.238)    Data  0.000 ( 0.018)    Loss 1.21243 (1.17996)  Acc@1  67.19 ( 69.58)   Acc@5  87.50 ( 88.76)
Validate: [ 0/99]   Time  3.398 ( 3.398)    Data  3.234 ( 3.234)    Loss 1.35944 (1.35944)  Acc@1  60.55 ( 60.55)   Acc@5  91.41 ( 91.41)
Validate: [99/99]   Time  0.050 ( 0.192)    Data  0.000 ( 0.078)    Loss 0.33242 (0.74631)  Acc@1  93.21 ( 79.52)   Acc@5  98.15 ( 95.25)
Acc@1 best:  79.52%
Epoch: [9][  0/295] Time  3.492 ( 3.492)    Data  3.076 ( 3.076)    Loss 1.11203 (1.11203)  Acc@1  71.48 ( 71.48)   Acc@5  88.67 ( 88.67)
Epoch: [9][100/295] Time  0.226 ( 0.257)    Data  0.000 ( 0.031)    Loss 1.14001 (1.14556)  Acc@1  72.27 ( 70.37)   Acc@5  88.28 ( 89.05)
Epoch: [9][200/295] Time  0.208 ( 0.237)    Data  0.000 ( 0.016)    Loss 1.11536 (1.14056)  Acc@1  74.61 ( 70.27)   Acc@5  89.06 ( 89.18)
Validate: [ 0/99]   Time  3.343 ( 3.343)    Data  3.128 ( 3.128)    Loss 1.61359 (1.61359)  Acc@1  50.39 ( 50.39)   Acc@5  87.50 ( 87.50)
Validate: [99/99]   Time  0.046 ( 0.193)    Data  0.000 ( 0.080)    Loss 0.55139 (0.71052)  Acc@1  83.95 ( 80.41)   Acc@5  96.30 ( 95.57)
Acc@1 best:  80.41%
Epoch: [10][  0/295]    Time  3.716 ( 3.716)    Data  3.442 ( 3.442)    Loss 1.06115 (1.06115)  Acc@1  71.09 ( 71.09)   Acc@5  92.19 ( 92.19)
Epoch: [10][100/295]    Time  0.221 ( 0.259)    Data  0.000 ( 0.034)    Loss 1.21677 (1.09780)  Acc@1  71.09 ( 71.23)   Acc@5  88.67 ( 89.84)
Epoch: [10][200/295]    Time  0.221 ( 0.238)    Data  0.000 ( 0.017)    Loss 1.13938 (1.10171)  Acc@1  71.48 ( 71.26)   Acc@5  89.45 ( 89.79)
Validate: [ 0/99]   Time  3.419 ( 3.419)    Data  3.302 ( 3.302)    Loss 1.67140 (1.67140)  Acc@1  48.83 ( 48.83)   Acc@5  90.62 ( 90.62)
Validate: [99/99]   Time  0.045 ( 0.195)    Data  0.000 ( 0.086)    Loss 0.39122 (0.68643)  Acc@1  89.51 ( 80.79)   Acc@5  96.30 ( 95.77)
Acc@1 best:  80.79%
Epoch: [11][  0/295]    Time  4.034 ( 4.034)    Data  3.722 ( 3.722)    Loss 1.17213 (1.17213)  Acc@1  68.36 ( 68.36)   Acc@5  88.67 ( 88.67)
Epoch: [11][100/295]    Time  0.227 ( 0.260)    Data  0.000 ( 0.037)    Loss 1.04260 (1.06715)  Acc@1  73.44 ( 72.14)   Acc@5  91.02 ( 90.36)
Epoch: [11][200/295]    Time  0.217 ( 0.239)    Data  0.000 ( 0.019)    Loss 0.92882 (1.06389)  Acc@1  73.44 ( 72.08)   Acc@5  93.36 ( 90.35)
Validate: [ 0/99]   Time  3.298 ( 3.298)    Data  3.137 ( 3.137)    Loss 1.67171 (1.67171)  Acc@1  49.61 ( 49.61)   Acc@5  87.50 ( 87.50)
Validate: [99/99]   Time  0.045 ( 0.192)    Data  0.000 ( 0.079)    Loss 0.39672 (0.69628)  Acc@1  90.74 ( 80.60)   Acc@5  96.30 ( 95.50)
Epoch: [12][  0/295]    Time  3.797 ( 3.797)    Data  3.505 ( 3.505)    Loss 1.05308 (1.05308)  Acc@1  73.83 ( 73.83)   Acc@5  91.80 ( 91.80)
Epoch: [12][100/295]    Time  0.235 ( 0.260)    Data  0.000 ( 0.035)    Loss 1.02594 (1.04648)  Acc@1  75.00 ( 72.68)   Acc@5  91.02 ( 90.42)
Epoch: [12][200/295]    Time  0.216 ( 0.238)    Data  0.000 ( 0.018)    Loss 0.91116 (1.04970)  Acc@1  78.91 ( 72.55)   Acc@5  91.02 ( 90.43)
Validate: [ 0/99]   Time  3.544 ( 3.544)    Data  3.428 ( 3.428)    Loss 1.64682 (1.64682)  Acc@1  49.61 ( 49.61)   Acc@5  89.84 ( 89.84)
Validate: [99/99]   Time  0.045 ( 0.192)    Data  0.000 ( 0.083)    Loss 0.51127 (0.67992)  Acc@1  83.95 ( 81.04)   Acc@5  96.91 ( 95.89)
Acc@1 best:  81.04%
Epoch: [13][  0/295]    Time  3.172 ( 3.172)    Data  2.868 ( 2.868)    Loss 1.08406 (1.08406)  Acc@1  73.05 ( 73.05)   Acc@5  88.28 ( 88.28)
Epoch: [13][100/295]    Time  0.209 ( 0.265)    Data  0.000 ( 0.049)    Loss 1.20278 (1.01792)  Acc@1  64.84 ( 73.39)   Acc@5  88.28 ( 90.80)
Epoch: [13][200/295]    Time  0.207 ( 0.240)    Data  0.000 ( 0.024)    Loss 1.02603 (1.02425)  Acc@1  73.05 ( 73.35)   Acc@5  91.02 ( 90.69)
Validate: [ 0/99]   Time  3.512 ( 3.512)    Data  3.347 ( 3.347)    Loss 1.56069 (1.56069)  Acc@1  51.56 ( 51.56)   Acc@5  89.84 ( 89.84)
Validate: [99/99]   Time  0.047 ( 0.195)    Data  0.000 ( 0.084)    Loss 0.35583 (0.65610)  Acc@1  90.74 ( 81.73)   Acc@5  97.53 ( 96.00)
Acc@1 best:  81.73%
Epoch: [14][  0/295]    Time  4.015 ( 4.015)    Data  3.705 ( 3.705)    Loss 1.00000 (1.00000)  Acc@1  73.83 ( 73.83)   Acc@5  92.58 ( 92.58)
Epoch: [14][100/295]    Time  0.208 ( 0.263)    Data  0.000 ( 0.042)    Loss 0.91323 (0.99449)  Acc@1  79.69 ( 73.86)   Acc@5  91.80 ( 91.06)
Epoch: [14][200/295]    Time  0.221 ( 0.240)    Data  0.000 ( 0.021)    Loss 1.03151 (1.00285)  Acc@1  72.66 ( 73.57)   Acc@5  91.41 ( 91.06)
Validate: [ 0/99]   Time  3.435 ( 3.435)    Data  3.302 ( 3.302)    Loss 1.57274 (1.57274)  Acc@1  50.39 ( 50.39)   Acc@5  90.23 ( 90.23)
Validate: [99/99]   Time  0.045 ( 0.193)    Data  0.000 ( 0.081)    Loss 0.43177 (0.65553)  Acc@1  89.51 ( 81.72)   Acc@5  97.53 ( 96.01)
Epoch: [15][  0/295]    Time  3.268 ( 3.268)    Data  2.989 ( 2.989)    Loss 1.04077 (1.04077)  Acc@1  73.05 ( 73.05)   Acc@5  90.62 ( 90.62)
Epoch: [15][100/295]    Time  0.207 ( 0.262)    Data  0.000 ( 0.042)    Loss 0.96719 (0.95679)  Acc@1  74.61 ( 74.88)   Acc@5  91.02 ( 91.65)
Epoch: [15][200/295]    Time  0.212 ( 0.239)    Data  0.000 ( 0.021)    Loss 1.13013 (0.96920)  Acc@1  75.00 ( 74.49)   Acc@5  89.45 ( 91.52)
Validate: [ 0/99]   Time  4.454 ( 4.454)    Data  4.350 ( 4.350)    Loss 1.33401 (1.33401)  Acc@1  58.98 ( 58.98)   Acc@5  95.31 ( 95.31)
Validate: [99/99]   Time  0.045 ( 0.191)    Data  0.000 ( 0.072)    Loss 0.43797 (0.65119)  Acc@1  87.65 ( 81.86)   Acc@5  96.30 ( 95.94)
Acc@1 best:  81.86%
Epoch: [16][  0/295]    Time  4.198 ( 4.198)    Data  3.869 ( 3.869)    Loss 1.04371 (1.04371)  Acc@1  71.88 ( 71.88)   Acc@5  89.45 ( 89.45)
Epoch: [16][100/295]    Time  0.208 ( 0.265)    Data  0.000 ( 0.046)    Loss 1.07553 (0.94283)  Acc@1  71.48 ( 74.90)   Acc@5  90.62 ( 91.77)
Epoch: [16][200/295]    Time  0.229 ( 0.241)    Data  0.000 ( 0.023)    Loss 0.99231 (0.95556)  Acc@1  73.44 ( 74.66)   Acc@5  89.06 ( 91.51)
Validate: [ 0/99]   Time  3.281 ( 3.281)    Data  3.164 ( 3.164)    Loss 1.39466 (1.39466)  Acc@1  57.81 ( 57.81)   Acc@5  91.41 ( 91.41)
Validate: [99/99]   Time  0.045 ( 0.193)    Data  0.000 ( 0.080)    Loss 0.47298 (0.62990)  Acc@1  87.65 ( 82.44)   Acc@5  97.53 ( 96.20)
Acc@1 best:  82.44%
Epoch: [17][  0/295]    Time  3.544 ( 3.544)    Data  3.221 ( 3.221)    Loss 0.91209 (0.91209)  Acc@1  75.78 ( 75.78)   Acc@5  93.36 ( 93.36)
Epoch: [17][100/295]    Time  0.207 ( 0.259)    Data  0.000 ( 0.034)    Loss 0.94960 (0.92614)  Acc@1  76.56 ( 75.72)   Acc@5  92.58 ( 91.65)
Epoch: [17][200/295]    Time  0.212 ( 0.237)    Data  0.000 ( 0.017)    Loss 0.97062 (0.93118)  Acc@1  70.70 ( 75.50)   Acc@5  93.36 ( 91.68)
Validate: [ 0/99]   Time  3.420 ( 3.420)    Data  3.313 ( 3.313)    Loss 1.25701 (1.25701)  Acc@1  63.67 ( 63.67)   Acc@5  92.58 ( 92.58)
Validate: [99/99]   Time  0.045 ( 0.193)    Data  0.000 ( 0.077)    Loss 0.39906 (0.63444)  Acc@1  88.27 ( 82.25)   Acc@5  96.30 ( 96.20)
Epoch: [18][  0/295]    Time  3.176 ( 3.176)    Data  2.875 ( 2.875)    Loss 0.77397 (0.77397)  Acc@1  79.69 ( 79.69)   Acc@5  94.14 ( 94.14)
Epoch: [18][100/295]    Time  0.209 ( 0.264)    Data  0.000 ( 0.046)    Loss 0.91430 (0.91938)  Acc@1  73.44 ( 75.61)   Acc@5  91.41 ( 92.05)
Epoch: [18][200/295]    Time  0.208 ( 0.241)    Data  0.000 ( 0.023)    Loss 0.78249 (0.91881)  Acc@1  78.52 ( 75.63)   Acc@5  93.36 ( 92.09)
Validate: [ 0/99]   Time  3.430 ( 3.430)    Data  3.305 ( 3.305)    Loss 1.38807 (1.38807)  Acc@1  56.64 ( 56.64)   Acc@5  92.19 ( 92.19)
Validate: [99/99]   Time  0.045 ( 0.195)    Data  0.000 ( 0.085)    Loss 0.31395 (0.63287)  Acc@1  91.36 ( 82.14)   Acc@5  98.15 ( 96.19)
Epoch: [19][  0/295]    Time  3.370 ( 3.370)    Data  3.057 ( 3.057)    Loss 0.83570 (0.83570)  Acc@1  78.12 ( 78.12)   Acc@5  94.14 ( 94.14)
Epoch: [19][100/295]    Time  0.208 ( 0.262)    Data  0.000 ( 0.041)    Loss 1.02164 (0.90026)  Acc@1  70.70 ( 76.16)   Acc@5  89.45 ( 92.18)
Epoch: [19][200/295]    Time  0.214 ( 0.239)    Data  0.000 ( 0.021)    Loss 1.09468 (0.90342)  Acc@1  70.70 ( 76.16)   Acc@5  88.67 ( 92.09)
Validate: [ 0/99]   Time  3.349 ( 3.349)    Data  3.238 ( 3.238)    Loss 1.14798 (1.14798)  Acc@1  64.45 ( 64.45)   Acc@5  94.92 ( 94.92)
Validate: [99/99]   Time  0.045 ( 0.193)    Data  0.000 ( 0.079)    Loss 0.46656 (0.61852)  Acc@1  85.80 ( 82.64)   Acc@5  95.06 ( 96.24)
Acc@1 best:  82.64%
Epoch: [20][  0/295]    Time  3.411 ( 3.411)    Data  3.142 ( 3.142)    Loss 0.93511 (0.93511)  Acc@1  75.39 ( 75.39)   Acc@5  89.45 ( 89.45)
Epoch: [20][100/295]    Time  0.214 ( 0.261)    Data  0.000 ( 0.035)    Loss 0.94157 (0.88082)  Acc@1  73.44 ( 76.56)   Acc@5  91.02 ( 92.40)
Epoch: [20][200/295]    Time  0.208 ( 0.240)    Data  0.000 ( 0.018)    Loss 0.99242 (0.88786)  Acc@1  73.83 ( 76.41)   Acc@5  90.62 ( 92.30)
Validate: [ 0/99]   Time  3.247 ( 3.247)    Data  3.055 ( 3.055)    Loss 1.37191 (1.37191)  Acc@1  58.20 ( 58.20)   Acc@5  92.58 ( 92.58)
Validate: [99/99]   Time  0.047 ( 0.195)    Data  0.000 ( 0.080)    Loss 0.38036 (0.62255)  Acc@1  88.89 ( 82.51)   Acc@5  98.15 ( 96.36)
elapsed time = 0h 29m 11s