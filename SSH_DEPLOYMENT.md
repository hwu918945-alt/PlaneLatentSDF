# 独立仓库 Slurm 部署

先按根README安装，并运行prepare-run。每个run包含自己的 `runtime/plane_auto_occ/`、cache、config、初始面积池及source_lock。Slurm只从该冻结runtime执行，不依赖任何 PlaneLatentSDF 工作树。

```bash
python -m plane_auto_occ prepare-run \
  --case /absolute/path/002_ED.npz --observed 0,3,6 \
  --output /absolute/path/PlaneAutoOccupancy/runs/new_002_ED
export TASK_EXPERIMENT=/absolute/path/PlaneAutoOccupancy/runs/new_002_ED
export TASK_PY=/absolute/path/cuda_python/bin/python
sbatch --job-name=plane_auto_occ_002_ED \
  --output="$TASK_EXPERIMENT/logs/auto_occ_%j.out" \
  --error="$TASK_EXPERIMENT/logs/auto_occ_%j.err" \
  scripts/single_case_job.sh
```

脚本默认分区aiaca40、qos4a40、1GPU、4CPU、24GB、4小时，可通过sbatch显式覆盖资源选项。TASK_CODE默认该run的runtime，TASK_PY默认python；建议明确提供既有CUDA Python路径。

先验证源码/cache/config/pool锁并运行原6项及修复6项测试，随后两步cold smoke，再新进程重置初始化和随机种子完成配置中的正式更新（默认2000）。只有显式提供prepare-run的 `--reference-initial` 才比较初始化，对照文件复制进run且被锁定。

观测索引由run锁读取，严格隐藏索引为剩余缓存层，case文件名不限定002_ED。每层采样、分块、loss和optimizer协议不因部署迁移改变。完成正式训练后才评估隐藏轮廓，并输出+5mm和+20mm评估、uncapped128³网格、事后验收；无早停或best checkpoint。

主要产物为 outputs/fit/history.jsonl、里程碑、动态面积池、final_eval、final_eval_margin20mm、myocardium.obj/json、posthoc_audit.json，以及GPU_SMOKE_COMPLETE/TRAIN_COMPLETE/RUN_COMPLETE。PIPELINE_FAILED仅记录执行失败。事后观测验收失败不会改写训练预算。

真实数据、权重、日志和派生mesh都存runs并被Git忽略；新clone只携带代码和文档。
