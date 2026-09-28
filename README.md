# CIFAR-10 Image Classifier

A small PyTorch project for classifying CIFAR-10 images with a convolutional neural network. It includes data exploration, training, evaluation, and plots of the results.


### Model and data

The model uses two convolutional layers, each followed by ReLU and max pooling, then two fully connected layers. It takes a 32 × 32 RGB image and returns scores for 10 classes. Training uses Adam and cross-entropy loss, and currently runs on the CPU.

The 50,000 training images are split into 45,000 training and 5,000 validation samples. The separate 10,000-image test set is used for evaluation. Images are converted to tensors and normalized with a mean and standard deviation of 0.5 for each channel. CIFAR-10 is downloaded automatically to `data/` on the first run.

### Setup

The project has been run with Python 3.11. Run these commands from the project root:

```bash
conda create -n cifar10-image-classifier python=3.11
conda activate cifar10-image-classifier
python -m pip install -r requirements.txt
```

The file lists the packages used directly by the project, with versions from the local development environment. pip installs their dependencies automatically. Installation on other platforms has not been verified.

### Run

```bash
# Train and save the best validation checkpoints
python -m src.train

# Plot training and validation loss and accuracy
python -m src.plot_history

# Evaluate the checkpoint with the best validation accuracy
python -m src.evaluate

# Run the model output-shape test
python -m pytest -q
```

Run training before evaluation: model weights are not included in the repository. Evaluation prints overall and per-class test accuracy and common classification errors. It also saves a confusion matrix and a grid of misclassified images; the printed metrics are not saved to a separate file.

For data exploration, open `notebooks/01_eda.ipynb` in VS Code or Jupyter and select the project environment. Notebook support is optional; install `ipykernel` in that environment if needed. The notebook shows class counts and sample images.

### Experiments

Edit `src/config.py` before running an experiment:

| Setting | Current value | Purpose |
| --- | --- | --- |
| `EXPERIMENT_NAME` | `"baseline"` | Names the output directories |
| `BATCH_SIZE` | `128` | Batch size for all data loaders |
| `LEARNING_RATE` | `1e-3` | Adam learning rate |
| `NUM_EPOCHS` | `10` | Number of training epochs |
| `NUM_CLASSES` | `10` | Number of model output classes |
| `SEED` | `42` | Seed for the data split, model initialization, and training-data shuffling |

`USE_AUGMENTATION` and `WEIGHT_DECAY` are defined in the config but are not yet used by the training pipeline. Changing them currently has no effect. The seed is used for the data split, model initialization, and training-data shuffling in the current CPU pipeline.

Use a different experiment name to keep separate results. Keep the same name when training, plotting, and evaluating one experiment. Reusing a name writes into the existing directories and can overwrite earlier files. The run configuration is not saved alongside the results.

### Files and outputs

```text
src/
  config.py          Experiment settings
  data.py            Dataset, transforms, and data loaders
  model.py           CNN definition
  train.py           Training and checkpoint saving
  evaluate.py        Test evaluation and error plots
  plot_history.py    Loss and accuracy curves
notebooks/
  01_eda.ipynb       Data exploration
tests/
  test_model.py     Model output-shape test
checkpoints/<experiment>/
  best_accuracy_model.pth
  best_loss_model.pth
outputs/metrics/<experiment>/history.csv
outputs/figures/<experiment>/
  loss_curve.png
  accuracy_curve.png
  confusion_matrix.png
  misclassified_examples.png
outputs/figures/sample_images.png
```

The EDA sample image is shared across experiments. Dataset files, model weights, Python caches, and local editor settings are excluded by `.gitignore`.

### Saved baseline results

The included `outputs/metrics/baseline/history.csv` contains **10 epochs**, matching the current `NUM_EPOCHS` setting. The best validation accuracy is **72.20% at epoch 10**. Validation loss reaches its minimum of **0.8271 at epoch 6**, then rises overall while training loss continues to fall, suggesting overfitting. These figures describe validation performance; test accuracy is printed separately by `src.evaluate`.



这是一个使用 PyTorch 实现的 CIFAR-10 图像分类项目。模型是一个小型卷积神经网络，项目包含数据探索、训练、评估和结果绘图。

### 模型与数据

模型包含两个卷积层，每层后接 ReLU 和最大池化，最后接两个全连接层。输入为 32 × 32 的 RGB 图片，输出为 10 个类别的分数。训练使用 Adam 优化器和交叉熵损失，目前在 CPU 上运行。

50,000 张训练图片划分为 45,000 张训练样本和 5,000 张验证样本，另有 10,000 张测试图片用于评估。图片转为张量后，各通道使用均值 0.5、标准差 0.5 进行标准化。首次运行时，CIFAR-10 会自动下载到 `data/`。

### 安装

项目已在 Python 3.11 环境下运行。请在项目根目录执行：

```bash
conda create -n cifar10-image-classifier python=3.11
conda activate cifar10-image-classifier
python -m pip install -r requirements.txt
```

`requirements.txt` 只列出项目直接使用的包，版本来自本地开发环境。它们所需的间接依赖会由 pip 自动安装。尚未验证其他平台上的安装情况。

### 运行

```bash
# 训练并保存验证集表现最好的模型
python -m src.train

# 绘制训练集和验证集的损失、准确率曲线
python -m src.plot_history

# 使用验证集准确率最高的模型进行测试集评估
python -m src.evaluate

# 运行模型输出尺寸测试
python -m pytest -q
```

仓库不包含模型权重，因此需要先训练，再评估。评估会在终端打印测试集总体准确率、各类别准确率和常见分类错误，并保存混淆矩阵与错误分类样本图。终端打印的指标目前不会单独保存为文件。

如需探索数据，可在 VS Code 或 Jupyter 中打开 `notebooks/01_eda.ipynb`，并选择项目环境。Notebook 支持是可选的；如有需要，请在该环境中安装 `ipykernel`。Notebook 展示各类别的样本数量和图片示例。

### 实验配置

运行实验前，修改 `src/config.py`：

| 配置 | 当前值 | 用途 |
| --- | --- | --- |
| `EXPERIMENT_NAME` | `"baseline"` | 指定输出目录名称 |
| `BATCH_SIZE` | `128` | 所有数据加载器的批次大小 |
| `LEARNING_RATE` | `1e-3` | Adam 学习率 |
| `NUM_EPOCHS` | `10` | 训练轮数 |
| `NUM_CLASSES` | `10` | 模型输出类别数 |
| `SEED` | `42` | 数据划分、模型初始化和训练数据打乱的随机种子 |

配置文件中的 `USE_AUGMENTATION` 和 `WEIGHT_DECAY` 尚未接入训练流程，修改它们目前不会产生效果。在当前 CPU 训练流程中，数据划分、模型初始化和训练数据打乱都使用了该随机种子。

不同实验请使用不同名称，以便保留各自的结果。同一次实验的训练、绘图和评估需要使用相同名称。重复使用一个名称会写入已有目录，可能覆盖之前的文件。目前不会随结果保存该次实验的配置。

### 文件与输出

```text
src/
  config.py          实验配置
  data.py            数据集、预处理和数据加载器
  model.py           CNN 模型
  train.py           训练与模型保存
  evaluate.py        测试集评估与错误分析图
  plot_history.py    损失和准确率曲线
notebooks/
  01_eda.ipynb       数据探索
tests/
  test_model.py     模型输出尺寸测试
checkpoints/<experiment>/
  best_accuracy_model.pth
  best_loss_model.pth
outputs/metrics/<experiment>/history.csv
outputs/figures/<experiment>/
  loss_curve.png
  accuracy_curve.png
  confusion_matrix.png
  misclassified_examples.png
outputs/figures/sample_images.png
```

EDA 样本图由所有实验共用。数据集、模型权重、Python 缓存和本地编辑器配置已通过 `.gitignore` 排除。

### 已保存的 baseline 结果

仓库中的 `outputs/metrics/baseline/history.csv` 包含 **10 轮**记录，与当前 `NUM_EPOCHS` 设置一致。最高验证集准确率为 **72.20%，出现在第 10 轮**。验证损失在**第 6 轮达到最低值 0.8271**，之后总体上升，而训练损失继续下降，说明模型出现了过拟合迹象。这些数值反映的是验证集表现，测试集准确率由 `src.evaluate` 单独打印。
