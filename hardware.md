hardware
====
機械学習用のPCのハードウェア構成とハードウェア依存のエラー解消方法  

## ハードウェア構成
| 機器名 | 型番 |  
| :---- | :--- |  
| マザーボード |  ROG MAXIMUS XI HERO (WI-FI) |  
| CPU |  Corei9-9900K |  
| CPUファン |  NH-U12A |  
| メモリ | CT2K16G4DFD8266 x2 |  
| グラフィック | TURBO-RTX2080TI-11G x2 |  
| SSD | 860 EVO MZ-76E500B/IT |  
| HDD |  WD80EFAX |  
| 電源 | HX1200i CP-9020070-JP |  

## エラー解消方法
### 突然電源が落ちる
#### ハードウェアの問題  
- 電源が半挿しになってないか確認  
  - 意外と多い
- CPU補助電源を挿してるか確認  
  - 動くが, 1時間もしないうちに落ちるとかはこれの可能性あり  

#### ソフトウェアの問題  
- BIOSのCPU Turbo Modeを切る  
  - https://github.com/tensorflow/tensorflow/issues/8858#issuecomment-580906512  
