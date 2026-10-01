# CIFAR-10 Image Classifier

[English](#english) · [中文](#中文)

## English

A PyTorch learning project that follows a small CNN from a two-layer baseline to a three-layer model with data augmentation, Batch Normalization and learning-rate scheduling. It includes training, epoch-level resume, evaluation, error analysis and single-image prediction.

The three final 60-epoch runs achieve **84.22% ± 0.33 percentage points test accuracy** across seeds 42, 43 and 44. This is the mean and sample standard deviation of three separately evaluated models, not an ensemble. Model weights and the dataset are not included in Git.

### Model and data

The current model is trained from scratch. With both `USE_BATCH_NORM` and `USE_EXTRA_CONV` enabled, its structure is:

```text
RGB image [B, 3, 32, 32]
  → Conv 3×3, 3→32 → BatchNorm → ReLU → MaxPool 2×2
  → Conv 3×3, 32→64 → BatchNorm → ReLU
  → Conv 3×3, 64→64 → BatchNorm → ReLU → MaxPool 2×2
  → Flatten [B, 4096] → Linear 128 → ReLU → Linear 10
  → logits [B, 10]
```

All convolutions use stride 1 and padding 1. `USE_EXTRA_CONV=False` removes the third convolution; `USE_BATCH_NORM=False` replaces BN layers with identity operations. Training uses Adam and `CrossEntropyLoss` directly on logits. Single-image prediction applies softmax to report a class score.

The ten classes are airplane, automobile, bird, cat, deer, dog, frog, horse, ship and truck. CIFAR-10 provides 50,000 training images and 10,000 test images. This project splits the training set into **45,000 train / 5,000 validation**, with `SPLIT_SEED=42`. `SEED` controls training randomness and is varied across repeated runs. Fixing seeds does not guarantee identical results across devices or library versions.

All images use `ToTensor()` and channel-wise normalization with mean `(0.5, 0.5, 0.5)` and standard deviation `(0.5, 0.5, 0.5)`. When enabled, training augmentation adds `RandomCrop(32, padding=4)` and `RandomHorizontalFlip(p=0.5)`. Validation and test images have no random augmentation. The split indices are shared between separate training and validation dataset objects, so validation never inherits training augmentation.

### Setup and run

Use Python 3.11 and run commands from the project root:

```bash
conda create -n cifar10-image-classifier python=3.11
conda activate cifar10-image-classifier
python -m pip install -r requirements.txt
```

The pinned requirements record the local development environment. Installation on every OS/GPU combination has not been verified. CUDA requires a compatible CUDA-enabled PyTorch installation. `DEVICE="auto"` selects CUDA first, then Apple MPS, then CPU; an explicitly requested unavailable device raises an error. CIFAR-10 downloads into `data/` on the first training or evaluation run.

1. Edit [src/config.py](src/config.py). For a **new** run, choose an unused `EXPERIMENT_NAME` and set `RESUME_TRAINING=False`. For example, use `cnn3_bn_aug_sched_s42_60ep_v2` with `SEED=42`. The checked-in configuration currently points to `mps_bn_extra_conv_seed44_60ep`.
2. Train, plot and evaluate with the same experiment name and model flags:

```bash
python -m src.train
python -m src.plot_history
python -m src.evaluate
```

Plotting reads the saved CSV, so it can also run after evaluation. Evaluation loads `best_accuracy_model.pth`, prints overall/per-class test accuracy and common errors, and saves a confusion matrix and misclassified-image grid. It does **not** automatically save `evaluation.txt`; the result files included here are captured console reports. To capture a new report, replace the example name with the exact configured experiment:

```bash
# macOS / Linux
python -m src.evaluate | tee outputs/metrics/cnn3_bn_aug_sched_s42_60ep_v2/evaluation.txt
```

```powershell
# Windows PowerShell
python -m src.evaluate | Tee-Object -FilePath outputs/metrics/cnn3_bn_aug_sched_s42_60ep_v2/evaluation.txt
```

Use the capture command instead of the plain evaluation command when a text report is needed. The metrics directory is created by training.

3. Predict an external image using the same model configuration and weights:

```bash
python -m src.predict "my_image.jpg"
```

Example output (the score depends on the image and checkpoint):

```text
Prediction: cat
Confidence: 88.3%
```

Prediction converts the image to RGB, resizes it to 32×32, applies the same normalization, adds a batch dimension and runs `model.eval()` with gradients disabled. It then applies softmax and selects the highest-scoring class. It does not load or download CIFAR-10. The score is not a calibrated probability of correctness, and the model always chooses one of the ten classes, even for unrelated images. Higher-resolution input is still reduced to 32×32.

4. Run the existing model-shape test:

```bash
python -m pytest -q
```

For data exploration, open `notebooks/01_eda.ipynb` in VS Code or Jupyter with the project environment. Install `ipykernel` separately if needed.

### Configuration and why it changed

| Setting / change | Current value | Reason and evidence |
| --- | --- | --- |
| Data augmentation | `USE_AUGMENTATION=True` | Random crops and flips expose the model to different positions and views. The early baseline reached 98.93% training accuracy but only 70.90% validation accuracy at epoch 40. Augmentation was introduced to improve generalization. |
| Learning-rate scheduler | `USE_LR_SCHEDULER=True` | Reduce the step size when validation loss stops improving, allowing finer updates later in training. The early paired three-seed comparison improved mean best validation accuracy by 0.38 pp. This was a modest observed gain. |
| Scheduler parameters | `mode="min"`, `factor=0.5`, `patience=3`, `min_lr=1e-5` | Monitor validation loss, halve the learning rate after more than three consecutive non-improving epochs, and stop reducing it at the floor. The scheduler runs after each validation epoch. |
| Epoch budget | `NUM_EPOCHS=60` | Ten epochs were insufficient for the augmented models. The three-layer model improved from 40 to 60 epochs across three seeds; the seed-42 extension to 80 epochs added only 0.10 pp to its best validation accuracy. Sixty is a measured budget choice, not a universal optimum. |
| Batch Normalization (BN) | `USE_BATCH_NORM=True` | Normalize intermediate convolutional activations to help optimization. Its observed mean gain in the two-layer MPS comparison was 0.53 pp, with variation across seeds. |
| Extra convolution | `USE_EXTRA_CONV=True` | Add another feature-extraction layer before the second pooling operation. This was the largest gain in the repeated MPS architecture comparisons: +4.56 pp at 40 epochs. |
| Batch size | `BATCH_SIZE=128` | Process 128 images per optimization step, balancing memory use and throughput. Batch size was held fixed, not tuned in a controlled comparison. There is no measured accuracy gain attributable to changing it. It is separate from BatchNorm. |
| Initial learning rate | `LEARNING_RATE=1e-3` | Starting step size for Adam, held fixed in the current configuration. The scheduler changes the effective rate during training. |
| Seeds | `SPLIT_SEED=42`, `SEED=42/43/44` | Keep the data partition fixed while checking training variability. The current config uses training seed 44. |
| Device / workers | `DEVICE="auto"`, `CUDA_NUM_WORKERS=2` | Use available acceleration. CUDA training uses worker processes and pinned memory; CPU/MPS training uses zero workers. Device changes are not treated as an accuracy improvement. |
| Weight decay | `WEIGHT_DECAY=0.0` | This config field currently is **not passed to Adam**. Changing it alone has no effect; no weight-decay benefit is claimed. |

### Recorded results

All validation figures below come from saved `history.csv` files. **Loss means validation cross-entropy at the epoch with the highest validation accuracy**, choosing the first epoch in a tie, as checkpoint selection does. It is not necessarily the minimum loss or the final-epoch loss. Accuracy changes use percentage points (pp); lower loss is better. Multi-seed summaries average each run's selected epoch and use sample standard deviation (`n−1`).

The early seed-42 progression was:

| Early experiment (seed 42, 40 epochs) | Best val accuracy | Corresponding val loss | Accuracy change |
| --- | ---: | ---: | ---: |
| [`baseline`](outputs/metrics/baseline/history.csv) | 72.20% | 0.9113 | — |
| [`Augmented_CNN_Experiment`](outputs/metrics/Augmented_CNN_Experiment/history.csv) | 77.66% | 0.6520 | +5.46 pp |
| [`augmentation_scheduler_40epochs`](outputs/metrics/augmentation_scheduler_40epochs/history.csv) | 78.16% | 0.6392 | +0.50 pp |

Across seeds 42/43/44, augmentation without a scheduler reached **78.49%** mean best validation accuracy and **0.6435** corresponding loss. With the scheduler, these were **78.87%** and **0.6288**: **+0.38 pp**, loss **−0.0146**. Individual accuracy gains were +0.50, +0.06 and +0.58 pp. These small-sample results do not establish a guaranteed improvement.

The later MPS comparisons were:

| Stage | Epochs | Best val accuracy, mean ± SD | Mean corresponding val loss | Accuracy change | Loss change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Two conv + augmentation + scheduler (MPS) | 40 | 78.72% ± 0.74 | 0.6249 | — | — |
| Add BatchNorm | 40 | 79.25% ± 0.50 | 0.6109 | +0.53 pp | -0.0140 |
| Add a third convolution | 40 | 83.81% ± 0.20 | 0.4783 | +4.56 pp | -0.1326 |
| Extend to 60 epochs | 60 | 84.83% ± 0.34 | 0.4581 | +1.02 pp | -0.0202 |

Rows compare successive stages within the MPS series. Do not add the early-series and MPS-series gains together: the execution environment and code evolved. Older runs lack complete saved configurations, so their labels follow the recorded experiment history and directory names. BN/extra-convolution flags were cross-checked against local weight keys; the final 60-epoch runs also have saved `run_config` metadata. The tables describe observed differences, not proof that each change helps every run.

For the three-layer BN model, the seed-42 learning curve also explains the epoch budget:

| Training budget | Best val accuracy so far | Best epoch | Corresponding val loss |
| ---: | ---: | ---: | ---: |
| 10 | 75.16% | 9 | 0.7280 |
| 20 | 79.46% | 17 | 0.5951 |
| 40 | 83.80% | 37 | 0.4832 |
| 60 | 84.44% | 52 | 0.4632 |
| 80 | 84.54% | 79 | 0.4614 |

These are prefix summaries of the [80-epoch history](outputs/metrics/mps_bn_extra_conv_seed42_80ep/history.csv), not five independent experiments. Its first 40 and 60 rows exactly match the saved seed-42 40/60-epoch histories. The improvement slows considerably after 60 epochs, and 80 epochs has only been checked for one seed.

Final 60-epoch test results, using each run's best-validation-accuracy weights:

| Seed | Best val accuracy | Corresponding val loss | Test accuracy |
| ---: | ---: | ---: | ---: |
| 42 | 84.44% | 0.4632 | [83.90%](outputs/metrics/mps_bn_extra_conv_seed42_60ep/evaluation.txt) |
| 43 | 85.06% | 0.4492 | [84.20%](outputs/metrics/mps_bn_extra_conv_seed43_60ep/evaluation.txt) |
| 44 | 85.00% | 0.4619 | [84.55%](outputs/metrics/mps_bn_extra_conv_seed44_60ep/evaluation.txt) |

**Mean test accuracy: 84.22% ± 0.33 pp.** Test loss is not recorded by the current evaluator and is therefore not reported. Checkpoints are selected by validation performance; use validation data for future tuning and reserve the test set for final comparisons. External photographs are a different evaluation setting from CIFAR-10.

See the [experiment index](docs/experiments.md) for every saved run, exact epochs and links to its logs. The temporary runs are listed for traceability but excluded from the main comparisons.

### Pause and resume

Press `Ctrl+C` to stop training. Each completed epoch writes `last_checkpoint.pth`, including model weights, optimizer/scheduler state, history, best weights, relevant settings and random states. An interrupted partial epoch is discarded and rerun after resuming; interruption before the first saved epoch leaves nothing to resume.

Keep the experiment name and training settings unchanged, then set:

```python
RESUME_TRAINING = True
NUM_EPOCHS = 60  # Total target, not 60 additional epochs
```

```bash
python -m src.train
```

For example, resuming a 40-epoch run with a target of 60 trains epochs 41–60. Keep its original folder name even if it ends in `40ep`; the CSV records the actual length. A new name selects a different checkpoint directory. The loader rejects changes to saved training settings; the total epoch target may increase. Device changes are supported, but identical random sequences/results across devices are not guaranteed. Older experiments with only `best_*_model.pth` can be evaluated but cannot resume the full optimizer/scheduler state.

### Files, Git and artifact storage

Current paths (used by the code):

```text
src/
  config.py          Experiment settings
  data.py            Splits, transforms and data loaders
  device.py          CUDA / MPS / CPU selection
  model.py           CNN with optional BN and third convolution
  train.py           Training and epoch-level resume
  plot_history.py    Loss and accuracy curves
  evaluate.py        Test metrics and error analysis
  predict.py         Single-image inference
docs/experiments.md  Index of recorded runs
notebooks/01_eda.ipynb
tests/test_model.py
outputs/metrics/<experiment>/
  history.csv
  evaluation.txt     Optional captured console report
outputs/figures/<experiment>/
  loss_curve.png
  accuracy_curve.png
  confusion_matrix.png
  misclassified_examples.png
checkpoints/<experiment>/
  best_accuracy_model.pth  Evaluation and prediction weights
  best_loss_model.pth      Lowest-validation-loss weights
  last_checkpoint.pth     Full resume state for newer runs
```

`outputs/figures/sample_images.png` is shared EDA output. Configuration for newer runs is embedded in `last_checkpoint.pth`, not exported to a standalone JSON file. When sharing a best-weights file, include the matching config and experiment name so the recipient can reconstruct the model.

Recommended storage policy, without changing the current paths:

| Artifact | Git / local policy |
| --- | --- |
| Source, README, requirements, experiment index | Commit to Git. |
| Small `history.csv` files and final evaluation reports | Keep as reproducible evidence in Git. |
| Plots | Keep representative baseline/final plots in Git; archive redundant exploratory plots separately. |
| Final weights and full resume states | Keep the three final seed runs locally and in a backup. Keep `last_checkpoint.pth` for any run you may continue. Share weights separately with their configuration if needed. |
| Old/temporary runs | Archive each run's metrics, figures and checkpoints together outside the repository after checking the backup. Do not remove a run that still supports a documented comparison without updating its links. |
| Dataset and environments | Regenerate/download as needed; exclude from Git. |

The existing `.gitignore` already excludes `data/` and `checkpoints/`. It does not exclude `outputs/`. Adding ignore rules later will not untrack files already committed. No artifacts have been moved or deleted as part of this documentation update.

For new experiments, use descriptive names such as `cnn3_bn_aug_sched_s42_60ep_v2`, and record the actual epoch count in the index. An index plus an archive is sufficient at this project's current size. A future `runs/<experiment>/{metrics,figures,checkpoints}` layout would group each run in one place, but requires updating paths in training, plotting, evaluation and prediction together; it is not implemented here.

---

## 中文

这是一个使用 PyTorch 学习图像分类的项目：从两层 CNN 基线开始，逐步加入数据增强、Batch Normalization、学习率调度和第三个卷积层。项目包含训练、按完整 epoch 续训、评估、错误分析和单图片预测。

最终三个 60 轮模型在 seed 42、43、44 上的**测试准确率为 84.22% ± 0.33 个百分点**。这是三个独立模型的均值与样本标准差，不是模型集成结果。Git 不包含模型权重和数据集。

### 模型与数据

当前模型从头训练，未使用预训练权重。开启 `USE_BATCH_NORM` 和 `USE_EXTRA_CONV` 后，结构如下：

```text
RGB 图片 [B, 3, 32, 32]
  → 3×3 卷积，3→32 → BatchNorm → ReLU → 2×2 最大池化
  → 3×3 卷积，32→64 → BatchNorm → ReLU
  → 3×3 卷积，64→64 → BatchNorm → ReLU → 2×2 最大池化
  → 展平 [B, 4096] → 全连接 128 → ReLU → 全连接 10
  → logits [B, 10]
```

所有卷积的 stride 为 1、padding 为 1。关闭 `USE_EXTRA_CONV` 会移除第三个卷积层；关闭 `USE_BATCH_NORM` 后，BN 层被恒等操作替代。训练使用 Adam，将 logits 直接传给 `CrossEntropyLoss`；单图预测才通过 softmax 输出类别分数。

十个类别依次为飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船、卡车。CIFAR-10 自带 50,000 张训练图片和 10,000 张测试图片，项目将训练部分划分为 **45,000 张训练 / 5,000 张验证**。`SPLIT_SEED=42` 固定划分，`SEED` 控制训练随机性，用于多次重复实验。固定种子不能保证不同设备或库版本得到完全相同的结果。

所有图片均经过 `ToTensor()`，然后各通道使用均值 0.5、标准差 0.5 归一化。开启增强时，训练图片额外使用 `RandomCrop(32, padding=4)` 和 `RandomHorizontalFlip(p=0.5)`。验证集和测试集没有随机增强。训练与验证使用独立的数据集对象并共享划分索引，避免增强操作进入验证流程。

### 安装与运行

使用 Python 3.11，在项目根目录运行：

```bash
conda create -n cifar10-image-classifier python=3.11
conda activate cifar10-image-classifier
python -m pip install -r requirements.txt
```

依赖版本记录的是本地开发环境，尚未验证所有系统和 GPU 组合的安装情况。CUDA 需要与环境兼容且支持 CUDA 的 PyTorch。`DEVICE="auto"` 按 CUDA、Apple MPS、CPU 的顺序选择设备；显式指定不可用的设备会报错。首次训练或评估时，CIFAR-10 会下载到 `data/`。

1. 修改 [src/config.py](src/config.py)。**新实验**使用未占用的 `EXPERIMENT_NAME`，并设置 `RESUME_TRAINING=False`。例如名称 `cnn3_bn_aug_sched_s42_60ep_v2`，配合 `SEED=42`。当前保存的配置指向 `mps_bn_extra_conv_seed44_60ep`。
2. 保持实验名称和模型开关一致，依次训练、绘图、评估：

```bash
python -m src.train
python -m src.plot_history
python -m src.evaluate
```

绘图读取已有 CSV，因此也可以在评估后运行。评估加载 `best_accuracy_model.pth`，打印总体/各类别测试准确率与常见错误，并保存混淆矩阵和错误样本图。它**不会自动保存 `evaluation.txt`**，现有文本报告是保存下来的终端输出。需要保存报告时，用以下命令替代普通评估命令，并将示例名称替换为配置中的实验名称：

```bash
# macOS / Linux
python -m src.evaluate | tee outputs/metrics/cnn3_bn_aug_sched_s42_60ep_v2/evaluation.txt
```

```powershell
# Windows PowerShell
python -m src.evaluate | Tee-Object -FilePath outputs/metrics/cnn3_bn_aug_sched_s42_60ep_v2/evaluation.txt
```

metrics 目录由训练流程创建，不需要为了保存文本再重复评估一次。

3. 使用相同的模型配置和权重预测外部图片：

```bash
python -m src.predict "my_image.jpg"
```

输出示例，实际分数随图片和权重变化：

```text
Prediction: cat
Confidence: 88.3%
```

预测入口将图片转为 RGB、缩放至 32×32、做相同归一化并增加 batch 维度，再通过 `model.eval()` 关闭训练行为，在禁用梯度的情况下推理，最后用 softmax 选取分数最高的类别。这个入口不加载或下载 CIFAR-10。分数不是经过校准的正确概率；即使输入其他物体，也会从十个类别中选一个。高清图片最终仍缩小到 32×32。

4. 运行现有模型输出尺寸测试：

```bash
python -m pytest -q
```

数据探索可使用 VS Code 或 Jupyter 打开 `notebooks/01_eda.ipynb`，选择项目环境，按需单独安装 `ipykernel`。

### 配置及每一步修改的原因

| 配置 / 改动 | 当前值 | 原因与实验依据 |
| --- | --- | --- |
| 数据增强 aug | `USE_AUGMENTATION=True` | 随机裁剪、翻转让模型接触不同位置与视角。早期 baseline 第 40 轮训练准确率达到 98.93%，验证准确率却只有 70.90%，因此加入增强改善泛化。 |
| 学习率调度 scheduler | `USE_LR_SCHEDULER=True` | 验证 loss 停滞时缩小更新步长，让后期训练更细致。早期三个种子的成对比较中，平均最佳验证准确率提升 0.38 个百分点，属于较小的实测收益。 |
| Scheduler 参数 | `mode="min"`、`factor=0.5`、`patience=3`、`min_lr=1e-5` | 监控验证 loss；连续未改善轮数超过 3 后将学习率减半，最低降至设定下限。每轮验证结束后调用调度器。 |
| 训练轮数 epoch | `NUM_EPOCHS=60` | 增强后的模型在 10 轮时尚未训练充分。三层模型从 40 延长至 60 轮，在三个种子上都有改善；seed 42 从 60 延长至 80 轮，最佳验证准确率只再提高 0.10 个百分点。因此保留 60 轮作为当前训练预算，不代表它对所有模型都最优。 |
| 批归一化 BN | `USE_BATCH_NORM=True` | 归一化卷积层的中间特征，帮助优化。两层模型的 MPS 对照中，平均改善为 0.53 个百分点，不同种子收益不同。 |
| 第三个卷积层 | `USE_EXTRA_CONV=True` | 在第二次池化前多做一层特征提取。它是 MPS 架构对照中收益最大的改动：40 轮时平均提升 4.56 个百分点。 |
| 批次大小 batch size | `BATCH_SIZE=128` | 每次优化更新处理 128 张图片，兼顾内存占用与吞吐量。本项目没有对不同 batch size 做受控比较，因此不能声称增大 batch 带来准确率提升。它与 BatchNorm 是两个不同概念。 |
| 初始学习率 | `LEARNING_RATE=1e-3` | Adam 的起始更新步长，当前配置中保持固定；实际训练中的学习率由 scheduler 调整。 |
| 随机种子 | `SPLIT_SEED=42`、`SEED=42/43/44` | 固定数据划分，改变训练种子，观察结果波动。当前配置的训练种子为 44。 |
| 设备 / 加载进程 | `DEVICE="auto"`、`CUDA_NUM_WORKERS=2` | 使用可用设备加速。CUDA 训练启用加载进程与 pinned memory；CPU/MPS 使用零个加载子进程。切换设备不作为准确率改善项。 |
| 权重衰减 | `WEIGHT_DECAY=0.0` | 该字段目前**没有传给 Adam**，单独修改它不会生效，因此不报告 weight decay 的收益。 |

### 实验结果

下表验证指标来自保存的 `history.csv`。**loss 统一取“验证准确率最高那一轮”的验证交叉熵**；准确率相同时取第一次达到该值的轮次，与保存最佳准确率权重的逻辑一致。它不一定是最低 loss，也不一定是最后一轮 loss。准确率变化使用“百分点”，loss 越低越好。多种子结果先分别选出每次实验的最佳轮次，再计算均值与样本标准差（分母为 `n−1`）。

早期 seed 42 的实验过程：

| 早期实验（seed 42，40 轮） | 最佳验证准确率 | 对应验证 loss | 准确率变化 |
| --- | ---: | ---: | ---: |
| [`baseline`](outputs/metrics/baseline/history.csv) | 72.20% | 0.9113 | — |
| [`Augmented_CNN_Experiment`](outputs/metrics/Augmented_CNN_Experiment/history.csv) | 77.66% | 0.6520 | +5.46 个百分点 |
| [`augmentation_scheduler_40epochs`](outputs/metrics/augmentation_scheduler_40epochs/history.csv) | 78.16% | 0.6392 | +0.50 个百分点 |

在 seed 42/43/44 上，增强但不使用 scheduler 的平均最佳验证准确率为 **78.49%**，对应 loss 为 **0.6435**；加入 scheduler 后为 **78.87%**、**0.6288**，即 **+0.38 个百分点、loss −0.0146**。各个种子的准确率变化分别为 +0.50、+0.06、+0.58 个百分点。三个样本的结果不足以保证未来实验一定改善。

后续 MPS 实验对照：

| 阶段 | 轮数 | 最佳验证准确率，均值 ± 标准差 | 对应验证 loss 均值 | 准确率变化 | loss 变化 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 两层卷积 + 数据增强 + scheduler（MPS） | 40 | 78.72% ± 0.74 | 0.6249 | — | — |
| 加入 BatchNorm | 40 | 79.25% ± 0.50 | 0.6109 | +0.53 个百分点 | -0.0140 |
| 增加第三个卷积层 | 40 | 83.81% ± 0.20 | 0.4783 | +4.56 个百分点 | -0.1326 |
| 延长到 60 轮 | 60 | 84.83% ± 0.34 | 0.4581 | +1.02 个百分点 | -0.0202 |

每行与 MPS 系列的上一阶段比较。早期系列与 MPS 系列之间，运行环境和代码都发生过变化，因此不能把两组提升简单累加。旧实验没有完整的配置快照，其标签依据已有实验记录与目录名；BN 和额外卷积开关已通过本地权重键核对，最终三个 60 轮实验还有 `run_config` 元数据。表格描述实测差异，不代表每一步对所有训练都有效。

三层 BN 模型的 seed 42 曲线也说明了为什么调整 epoch：

| 训练预算 | 截至该轮的最佳验证准确率 | 最佳轮数 | 对应验证 loss |
| ---: | ---: | ---: | ---: |
| 10 | 75.16% | 9 | 0.7280 |
| 20 | 79.46% | 17 | 0.5951 |
| 40 | 83.80% | 37 | 0.4832 |
| 60 | 84.44% | 52 | 0.4632 |
| 80 | 84.54% | 79 | 0.4614 |

这些是同一份 [80 轮日志](outputs/metrics/mps_bn_extra_conv_seed42_80ep/history.csv) 的前缀统计，不是五次独立实验。前 40、60 行分别与已保存的 seed 42 的 40/60 轮日志完全相同。60 轮后的收益明显减小，而且只有一个种子验证了 80 轮。

最终 60 轮测试结果，每次均评估该实验中验证准确率最高的权重：

| Seed | 最佳验证准确率 | 对应验证 loss | 测试准确率 |
| ---: | ---: | ---: | ---: |
| 42 | 84.44% | 0.4632 | [83.90%](outputs/metrics/mps_bn_extra_conv_seed42_60ep/evaluation.txt) |
| 43 | 85.06% | 0.4492 | [84.20%](outputs/metrics/mps_bn_extra_conv_seed43_60ep/evaluation.txt) |
| 44 | 85.00% | 0.4619 | [84.55%](outputs/metrics/mps_bn_extra_conv_seed44_60ep/evaluation.txt) |

**测试准确率均值：84.22% ± 0.33 个百分点。** 当前评估脚本没有记录测试 loss，因此不填写该指标。模型权重根据验证集选择；后续调参使用验证集，测试集保留用于最终比较。日常外部照片与 CIFAR-10 是不同的评估场景。

[实验索引](docs/experiments.md) 列出了所有已保存实验、具体轮次和日志链接。临时实验保留索引用于追溯，不参与主表比较。

### 暂停与断点续训

按 `Ctrl+C` 停止训练。每个完整 epoch 结束后会写入 `last_checkpoint.pth`，包含模型、优化器、scheduler、训练历史、最佳权重、相关配置与随机状态。中途被打断的 epoch 不保存进度，恢复后会重新运行；如果第一轮尚未保存，就没有可恢复的断点。

保持实验名称和训练配置不变，设置：

```python
RESUME_TRAINING = True
NUM_EPOCHS = 60  # 总目标轮数，不是额外再跑 60 轮
```

```bash
python -m src.train
```

例如，已完成 40 轮时将目标设为 60，会继续训练第 41–60 轮。即使原名称包含 `40ep`，续训时也保留原名，实际长度以 CSV 为准；换名会查找另一个目录。恢复流程检查已保存的训练配置，允许增加总轮数。可以更换设备，但不保证跨设备随机序列和结果完全相同。旧实验如果只有 `best_*_model.pth`，可以评估，但不能完整恢复优化器和 scheduler 状态。

### 文件、Git 与实验归档

代码使用的目录结构如下：

```text
src/
  config.py          实验配置
  data.py            数据划分、预处理、加载器
  device.py          CUDA / MPS / CPU 选择
  model.py           可选 BN 和第三卷积层的 CNN
  train.py           训练与按完整 epoch 续训
  plot_history.py    loss 与 accuracy 曲线
  evaluate.py        测试指标与错误分析
  predict.py         单图片推理
docs/experiments.md  已记录实验的索引
notebooks/01_eda.ipynb
tests/test_model.py
outputs/metrics/<experiment>/
  history.csv
  evaluation.txt     可选的终端评估报告
outputs/figures/<experiment>/
  loss_curve.png
  accuracy_curve.png
  confusion_matrix.png
  misclassified_examples.png
checkpoints/<experiment>/
  best_accuracy_model.pth  用于评估和单图预测
  best_loss_model.pth      验证 loss 最低的权重
  last_checkpoint.pth     新实验的完整续训状态
```

`outputs/figures/sample_images.png` 是共用的 EDA 样本图。新实验的配置保存在 `last_checkpoint.pth` 内，尚未单独导出 JSON。分享最佳权重时需要同时提供对应配置和实验名称，让对方能够重建模型。

建议先采用以下整理方式，保留代码当前使用的路径：

| 文件类型 | Git / 本地保留方案 |
| --- | --- |
| 源码、README、依赖、实验索引 | 提交 Git。 |
| 小体积 `history.csv` 和最终评估报告 | 保留在 Git，作为结果依据。 |
| 图片 | Git 保留代表性的 baseline 和最终实验图片，重复的探索性图片另行归档。 |
| 最终权重与续训状态 | 三个最终种子的文件保留本地并备份；可能继续训练的实验保留 `last_checkpoint.pth`。需要分享权重时，附带配置单独分发。 |
| 旧实验 / 临时实验 | 将同一次实验的 metrics、figures、checkpoints 配套归档到仓库外，确认备份后再考虑移出工作目录。仍支撑文档比较的日志不能直接移走，否则要同步更新链接。 |
| 数据集与环境 | 按需重新下载或创建，不进入 Git。 |

现有 `.gitignore` 已排除 `data/` 和 `checkpoints/`，没有排除 `outputs/`。以后添加忽略规则，也不会自动取消已经提交文件的追踪。本次文档更新没有移动或删除任何实验文件。

新实验建议使用 `cnn3_bn_aug_sched_s42_60ep_v2` 这样的明确名称，并在索引中记录实际轮数。按当前规模，先使用“实验索引 + 仓库外归档”即可。未来若改为 `runs/<experiment>/{metrics,figures,checkpoints}`，需要同时修改训练、绘图、评估和预测的路径；当前尚未实施这个结构。
