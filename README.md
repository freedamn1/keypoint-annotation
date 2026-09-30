# 玉米 YOLO11：12 个关键点 + 2 个框

本项目用 `玉米数据0918/iPhone17` 的 LabelMe 标注训练两套模型：YOLO11-pose 预测中央植株的 `plant` 框与 12 个关键点，YOLO11-detect 预测 `ear` 框。两套模型的结果合并为同名 LabelMe JSON，包含 **12 个 `point`、1 个 `plant` 矩形、1 个 `ear` 矩形**，所有 shape 的 `group_id=100`。原始标注不会被脚本修改。

## 1. 数据约定与代码目录

- iPhone17：171 张 5712×4284 图片，每张有 `plant` 和 `ear` 框。109 张有完整 12 点，61 张有 9 点，1 张有 6 点。缺失点在 YOLO pose 标签中写为 `0 0 0`，已有点写为归一化 `x y 2`。
- 两套数据使用**同一批图片、同一 train/val/test 分组、同一固定裁图**：图像中央 85% 宽度、全部高度。当前 171 张的框和点均位于此范围内。固定裁图避免训练依赖人工框、推理却没有框。
- 每张源图只标一株，背景里仍有未标注邻株。固定裁图能降低干扰，但效果必须在独立 PM 样本上检验。
- 推理选取最接近画面中央的 `plant` 实例；`ear` 候选需落在该植株框内，并优先靠近预测的雌穗中部点。若 `ear` 模型没有合适检测，会由雌穗三点估出一个框，同时在终端明确报告。
- 输出 JSON 的 `imageData=null`，图片和 JSON 必须放在同一目录。12 点标签顺序由 [`pose_pipeline/core.py`](pose_pipeline/core.py) 的 `LABELS` 固定。

```text
yolo11_pose_pipeline/
  build_dataset.py       审计、分组切分、两套 YOLO 数据转换
  train.py               顺序训练 pose 与 ear 模型，并各自测试
  infer_labelme.py       两模型推理、同株配对、LabelMe 回写
  evaluate.py            点位像素误差及两个框的 IoU
  pose_pipeline/core.py  标签、ROI、格式规则
  tests/                 无 GPU 的转换与合并测试
```

`data/`、`runs/`、模型权重由 `.gitignore` 排除；代码可推送 GitHub，图像与标注单独上传服务器。

## 2. 本地转换并打包训练数据

以下命令在本地 PowerShell、从 `D:\project\课题组\yolo11_pose_pipeline` 运行：

```powershell
cd 'D:\project\课题组\yolo11_pose_pipeline'
python -m pytest -q
python build_dataset.py --source '..\玉米数据0918\iPhone17' --audit-only
python build_dataset.py --source '..\玉米数据0918\iPhone17' --output 'data\pose' --ear-output 'data\ear'
tar -czf 'data\training_bundle.tar.gz' -C data pose ear
```

`--audit-only` 只是程序检查重复标签、类型、坐标、group_id、裁图覆盖情况并打印数据统计；它不改文件，也不要求再人工逐张审阅。你修正后三张图片后，当前 171 张审计的 `warnings` 已为空。若缺少本地依赖，可运行 `pip install Pillow PyYAML numpy pytest`。`build_dataset.py` 在本地直接生成训练用裁图和标签；两个输出目录必须为空。归档仅包含训练所需的 `data/pose` 与 `data/ear`，不上传原始 LabelMe JSON 及其重复嵌入的 `imageData`。

当前工作区已经执行过上述转换与打包，`data/training_bundle.tar.gz` 约 1.01 GB，两套数据各有 171 张裁图和 171 个标签。**直接上传现成归档即可，不要重复运行构建命令**。只有原始标注或分组策略再次变化时，才重建到新的空目录并重新打包。

如果 PowerShell 环境没有 `tar`，也可用 `Compress-Archive -Path 'data\pose','data\ear' -DestinationPath 'data\training_bundle.zip'`，远端改用 `unzip` 解压。归档前可检查 `data/pose/splits.csv`，确认同一株或同一拍摄段没有跨 train/val/test；需要更精确分组时，先提供 `--groups-csv` 重建到新的空目录。

在 GitHub 创建空仓库，将占位地址改为自己的仓库后：

```powershell
git init
git add .
git status --short
git commit -m "Add maize pose and ear detection pipeline"
git branch -M main
git remote add origin https://github.com/你的账号/你的仓库.git
git push -u origin main
```

推送前确认 `git status --short` 未列出图片、标注、`data/` 或权重。源标注之后若又被修改，把转换数据重建到新的空目录；脚本不会覆盖旧目录。

## 3. 远端 GPU 环境

建议先用约 24 GB 显存的 NVIDIA GPU，`imgsz=1024, batch=4` 试跑。两套模型顺序训练，不需要同时驻留显存。16 GB 可先试 batch 2。请准备 Linux、Python 3.10/3.11、可用的 NVIDIA 驱动以及足够的数据与检查点磁盘空间。

```bash
nvidia-smi
git clone https://github.com/你的账号/你的仓库.git maize-pose
cd maize-pose
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

先按服务器 CUDA/驱动版本从 PyTorch 官方安装页选择匹配的 GPU 版 `torch`、`torchvision` 命令，再运行：

```bash
pip install -r requirements.txt
python -c "import torch, ultralytics; print('cuda:', torch.cuda.is_available(), 'ultralytics:', ultralytics.__version__)"
```

确认输出 `cuda: True`。首次加载 `yolo11s-pose.pt` 和 `yolo11s.pt` 可能需要联网下载；若服务器无法下载，先在本地获取并上传权重，再通过训练命令的 `--pose-model`、`--ear-model` 指定路径。

## 4. 上传已转换的训练数据

以下命令在**本地 PowerShell**运行，替换 USER、HOST 和远端路径：

```powershell
ssh USER@HOST 'mkdir -p ~/maize-pose/data'
scp 'D:\project\课题组\yolo11_pose_pipeline\data\training_bundle.tar.gz' USER@HOST:~/maize-pose/data/
```

然后在远端仓库运行：

```bash
cd ~/maize-pose
source .venv/bin/activate
tar -xzf data/training_bundle.tar.gz -C data
test -f data/pose/data.yaml
test -f data/ear/data.yaml
```

解压后目录应包含：

- `data/pose`：裁图、41 列 pose 标签、`data.yaml`、`metadata.json`、`splits.csv`。
- `data/ear`：同样的裁图、5 列 ear 检测标签、`data.yaml`。程序先尝试硬链接 pose 裁图以节省空间，文件系统不支持时复制。

本地转换时默认每 20 个连续图片编号组成一个分组，固定随机种子 42；当前数据约为 train 133、val 18、test 20 张。相邻块仍可能拍到同一株。若知道同株或同一拍摄段，可在本地建立覆盖全部 171 张的 `image,group` CSV，同株使用同一组，然后用 `--groups-csv data/groups.csv` 重建到**新的**输出目录，再打包。示例：

```csv
image,group
IMG_2866.jpeg,plant-a
IMG_2867.jpeg,plant-a
```

## 5. 训练与源设备测试

```bash
python train.py --pose-data data/pose/data.yaml --ear-data data/ear/data.yaml --task both --imgsz 1024 --batch 4 --epochs 150 --patience 30 --device 0
```

默认顺序训练 `yolo11s-pose.pt` 和 `yolo11s.pt`，分别用源设备 test 划分评估。若只需重训其中一个模型，使用 `--task pose` 或 `--task ear`。终端会输出两套实际 `best.pt` 路径；典型位置分别为 `runs/pose/maize-yolo11s-pose/weights/best.pt` 和 `runs/ear/maize-yolo11s-ear/weights/best.pt`，同名实验重跑时 Ultralytics 可能给目录追加编号。显存不足时先减 `--batch`，再考虑降低 `--imgsz`。

训练入口会在服务器上自动生成 `data.runtime.yaml`，把数据根路径改为解压后的实际绝对路径。因此本地 `data.yaml` 中的 Windows 路径不会影响远端训练；不需要在服务器重新运行 `build_dataset.py`。

调参先看 val；test 只用于最终检验。由于源图数量少，还需要以下 PM 独立评估。

## 6. iPhone17PM 推理与复核

把 PM JPEG 上传到远端 `data/pm/images/`，运行：

```bash
python infer_labelme.py --pose-weights runs/pose/maize-yolo11s-pose/weights/best.pt --ear-weights runs/ear/maize-yolo11s-ear/weights/best.pt --source data/pm/images --metadata data/pose/metadata.json --imgsz 1024 --device 0
```

成功预测的图像旁生成同名 JSON，含 12 点及 `plant`、`ear` 两个框。已有 JSON 默认跳过；明确要替换时才传 `--overwrite`。目前 PM 文件夹中已有 6184–6189 的试标 JSON，因此如果连同 JSON 一起上传，这些图会被跳过；如需看两框新结果，可仅上传 JPEG 到独立推理目录，或备份后使用 `--overwrite`。

程序打印低置信度点数、雌穗框是否由关键点估计，以及没有检出中央植株的图名。没有植株检测时不写伪造 JSON，需人工处理或重新推理。下载结果时将 JSON 与同名 JPEG 放在同一目录。LabelMe 中如框遮住点，可先在形状列表里隐藏框，调整点后再显示框。

## 7. 跨设备评估与下一轮训练

在首次训练前从 PM 中挑 20–30 张覆盖遮挡、密集邻株、不同叶位的图，**人工复核**后另存为 `data/pm_gold/*.json`，不参与首次训练。对相同图片推理后：

```bash
python evaluate.py --gold data/pm_gold --pred data/pm/images
```

评估输出 12 点在原图上的误差中位数、P90、25/50/100 像素内比例，另给 `plant` 与 `ear` 框的 IoU 中位数和漏预测图名。再记录每张图人工复核耗时，与从零标注比较。若耳框差，可单独改进 ear 检测模型；若关键点差，可提高 pose 的 `imgsz`、检查裁图及增加人工复核数据。未经复核的 AI 预测不应直接混入训练集。

## 8. 当前验证范围

本地已对真实 171 张运行只读审计，当前无警告；格式转换、坐标、两套训练标签、服务器路径重定位、合并 JSON 及评估逻辑通过无 GPU 测试。本机没有安装 Ultralytics，也没有远端 GPU，所以尚未实际跑模型训练或推理。远端环境就绪后，先用少量图片试跑并检查两框与 12 点，再处理全部 PM 图片。
