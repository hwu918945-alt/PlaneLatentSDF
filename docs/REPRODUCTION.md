# 当前方法复现

当前 SAX-only 心肌 shell 的远端数据准备、CUDA smoke、训练、续训、Slurm 和评价命令见 [SHELL_DEPLOYMENT.md](SHELL_DEPLOYMENT.md)。

原始 Epi 方法及初始化实验属于历史版本 `14ec357`；Bunny/balloon-dog 实验仍使用 `experiments/*/runtime` 内的独立源码快照。不要把旧 Epi 缓存或旧初始化配置传入新主训练入口。
