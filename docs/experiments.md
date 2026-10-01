# Experiment index / 实验索引

This index covers the 24 saved training histories present when the README was updated. `Best val` selects the first maximum validation accuracy; `Loss at best` is the loss from that same epoch. `Min loss` may come from a different epoch. Percentages are displayed as percentages, while CSV accuracies are fractions. Names identify the recorded runs; older CSVs do not contain full configurations. Checkpoint architecture keys were inspected separately, but weights do not record augmentation, optimizer settings or original device.

此索引覆盖 README 更新时保存的 24 份训练日志。`Best val` 取首次达到最高验证准确率的轮次，`Loss at best` 是同一轮的 loss；`Min loss` 可能来自另一轮。表中准确率显示为百分比，CSV 原值为小数。目录名标识已有实验，旧 CSV 不含完整配置。已另外检查权重中的模型结构键，但权重本身不记录增强、优化器配置或原始运行设备。

| Experiment / 实验 | Epochs / 轮数 | Best epoch / 最佳轮次 | Best val / 最佳验证准确率 | Loss at best / 对应 loss | Min loss / 最低 loss | Report / 评估报告 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| [`Augmented_CNN_Experiment`](../outputs/metrics/Augmented_CNN_Experiment/history.csv) | 40 | 35 | 77.66% | 0.6520 | 0.6411 | — |
| [`aug_seed43_no_scheduler`](../outputs/metrics/aug_seed43_no_scheduler/history.csv) | 40 | 37 | 78.66% | 0.6323 | 0.6323 | — |
| [`aug_seed43_scheduler`](../outputs/metrics/aug_seed43_scheduler/history.csv) | 40 | 37 | 78.72% | 0.6279 | 0.6193 | — |
| [`aug_seed44_no_scheduler`](../outputs/metrics/aug_seed44_no_scheduler/history.csv) | 40 | 37 | 79.16% | 0.6460 | 0.6226 | — |
| [`aug_seed44_scheduler`](../outputs/metrics/aug_seed44_scheduler/history.csv) | 40 | 40 | 79.74% | 0.6193 | 0.6193 | — |
| [`augmentation_scheduler_40epochs`](../outputs/metrics/augmentation_scheduler_40epochs/history.csv) | 40 | 39 | 78.16% | 0.6392 | 0.6325 | — |
| [`baseline`](../outputs/metrics/baseline/history.csv) | 40 | 10 | 72.20% | 0.9113 | 0.8271 | — |
| [`mps_bn_extra_conv_seed42_40ep`](../outputs/metrics/mps_bn_extra_conv_seed42_40ep/history.csv) | 40 | 37 | 83.80% | 0.4832 | 0.4789 | — |
| [`mps_bn_extra_conv_seed42_60ep`](../outputs/metrics/mps_bn_extra_conv_seed42_60ep/history.csv) | 60 | 52 | 84.44% | 0.4632 | 0.4614 | [test](../outputs/metrics/mps_bn_extra_conv_seed42_60ep/evaluation.txt) |
| [`mps_bn_extra_conv_seed42_80ep`](../outputs/metrics/mps_bn_extra_conv_seed42_80ep/history.csv) | 80 | 79 | 84.54% | 0.4614 | 0.4603 | — |
| [`mps_bn_extra_conv_seed43_40ep`](../outputs/metrics/mps_bn_extra_conv_seed43_40ep/history.csv) | 40 | 37 | 83.62% | 0.4772 | 0.4772 | — |
| [`mps_bn_extra_conv_seed43_60ep`](../outputs/metrics/mps_bn_extra_conv_seed43_60ep/history.csv) | 60 | 54 | 85.06% | 0.4492 | 0.4436 | [test](../outputs/metrics/mps_bn_extra_conv_seed43_60ep/evaluation.txt) |
| [`mps_bn_extra_conv_seed44_40ep`](../outputs/metrics/mps_bn_extra_conv_seed44_40ep/history.csv) | 40 | 40 | 84.02% | 0.4745 | 0.4745 | — |
| [`mps_bn_extra_conv_seed44_60ep`](../outputs/metrics/mps_bn_extra_conv_seed44_60ep/history.csv) | 60 | 59 | 85.00% | 0.4619 | 0.4609 | [test](../outputs/metrics/mps_bn_extra_conv_seed44_60ep/evaluation.txt) |
| [`mps_bn_seed42_10ep`](../outputs/metrics/mps_bn_seed42_10ep/history.csv) | 10 | 10 | 72.38% | 0.7926 | 0.7926 | — |
| [`mps_bn_seed42_40ep`](../outputs/metrics/mps_bn_seed42_40ep/history.csv) | 40 | 39 | 78.68% | 0.6255 | 0.6255 | — |
| [`mps_bn_seed43_40ep`](../outputs/metrics/mps_bn_seed43_40ep/history.csv) | 40 | 37 | 79.50% | 0.5896 | 0.5866 | — |
| [`mps_bn_seed44_40ep`](../outputs/metrics/mps_bn_seed44_40ep/history.csv) | 40 | 38 | 79.58% | 0.6174 | 0.6113 | — |
| [`mps_cnn_seed42_10ep`](../outputs/metrics/mps_cnn_seed42_10ep/history.csv) | 10 | 10 | 71.02% | 0.8136 | 0.8136 | — |
| [`mps_cnn_seed42_40ep`](../outputs/metrics/mps_cnn_seed42_40ep/history.csv) | 40 | 38 | 78.04% | 0.6349 | 0.6344 | — |
| [`mps_cnn_seed43_40ep`](../outputs/metrics/mps_cnn_seed43_40ep/history.csv) | 40 | 39 | 78.62% | 0.6296 | 0.6265 | — |
| [`mps_cnn_seed44_40ep`](../outputs/metrics/mps_cnn_seed44_40ep/history.csv) | 40 | 40 | 79.50% | 0.6102 | 0.6058 | — |
| [`temp_experiment`](../outputs/metrics/temp_experiment/history.csv) | 40 | 37 | 78.70% | 0.6462 | 0.6290 | — |
| [`temp_experiment1`](../outputs/metrics/temp_experiment1/history.csv) | 10 | 9 | 75.16% | 0.7280 | 0.7129 | — |

## Comparison groups / 对照分组

| Group / 分组 | Runs / 实验名称 |
| --- | --- |
| Early augmentation, no scheduler / 早期增强、不启用 scheduler | `Augmented_CNN_Experiment`, `aug_seed43_no_scheduler`, `aug_seed44_no_scheduler` |
| Early augmentation + scheduler / 早期增强与 scheduler | `augmentation_scheduler_40epochs`, `aug_seed43_scheduler`, `aug_seed44_scheduler` |
| MPS two-conv control / MPS 两层卷积对照 | `mps_cnn_seed{42,43,44}_40ep` |
| MPS two-conv + BN / MPS 两层卷积加 BN | `mps_bn_seed{42,43,44}_40ep` |
| MPS three-conv + BN / MPS 三层卷积加 BN | `mps_bn_extra_conv_seed{42,43,44}_40ep` |
| Final 60-epoch runs / 最终 60 轮实验 | `mps_bn_extra_conv_seed{42,43,44}_60ep` |
| Seed-42 budget extension / seed 42 延长训练 | `mps_bn_extra_conv_seed42_80ep` |
| Short checks / 短程检查 | `mps_cnn_seed42_10ep`, `mps_bn_seed42_10ep` |
| Temporary checks, excluded from aggregate results / 临时检查，不计入汇总 | `temp_experiment`, `temp_experiment1` |

Final 60-epoch checkpoint metadata confirms: batch size 128, initial Adam learning rate 0.001, split seed 42, augmentation/BN/extra convolution/scheduler enabled, scheduler mode `min`, factor 0.5, patience 3 and minimum LR 0.00001. Training seeds are 42, 43 and 44. The current training code does not pass `WEIGHT_DECAY` to the optimizer.

最终 60 轮 checkpoint 的配置元数据确认：batch size 为 128，Adam 初始学习率为 0.001，划分种子为 42，开启增强、BN、额外卷积和 scheduler；scheduler 模式为 `min`、factor 为 0.5、patience 为 3、最低学习率为 0.00001。训练种子为 42、43、44。当前训练代码未向优化器传入 `WEIGHT_DECAY`。

The seed-42 40/60-epoch logs match the first 40/60 rows of the 80-epoch log exactly. They are related training histories and must not be counted as independent repetitions. A missing report means no saved text report is available here; it does not prove evaluation was never run. Earlier console-only test results are not included in the README's verified test table.

seed 42 的 40/60 轮日志与 80 轮日志前 40/60 行完全一致，不能当作独立重复实验。缺少评估报告只表示这里没有保存文本，不代表从未评估。早期仅出现在终端中的测试结果没有计入 README 的已核对测试表。

Use [README.md](../README.md) for commands, interpretation and the proposed artifact-retention policy. Update this index when adding or archiving a run. Paths above reflect the current code layout; no migration has been performed.

运行命令、结果解释与归档建议见 [README.md](../README.md)。新增或归档实验后同步更新此索引。上述链接使用代码当前的目录结构，尚未进行目录迁移。
