# 玉米 YOLO11：12 个关键点 + 2 个框

本项目用 `玉米数据0918/iPhone17` 的 LabelMe 标注训练两套模型：YOLO11-pose 预测中央植株的 `plant` 框与 12 个关键点，YOLO11-detect 预测 `ear` 框。两套模型的结果合并为同名 LabelMe JSON，包含 **12 个 `point`、1 个 `plant` 矩形、1 个 `ear` 矩形**，所有 shape 的 `group_id=100`。原始标注不会被脚本修改。

## 1. 数据约定与代码目录

- iPhone17：171 张 5712×4284 图片，每张有 `plant` 和 `ear` 框。109 张有完整 12 点，61 张有 9 点，1 张有 6 点。缺失点在 YOLO pose 标签中写为 `0 0 0`，已有点写为归一化 `x y 2`。
- 两套数据使用**同一批图片、同一 train/val/test 划分、同一固定裁图**：图像中央 85% 宽度、全部高度。当前 171 张的框和点均位于此范围内。固定裁图避免训练依赖人工框、推理却没有框。
- 每张源图只标一株，背景里仍有未标注邻株。固定裁图能降低干扰，但效果必须在独立 PM 样本上检验。
- 推理选取最接近画面中央的 `plant` 实例；`ear` 候选需落在该植株框内，并优先靠近预测的雌穗中部点。若 `ear` 模型没有合适检测，会由雌穗三点估出一个框，同时在终端明确报告。
- 输出 JSON 的 `imageData=null`，图片和 JSON 必须放在同一目录。12 点标签顺序由 [`pose_pipeline/core.py`](pose_pipeline/core.py) 的 `LABELS` 固定。

```text
yolo11_pose_pipeline/
  build_dataset.py       审计、逐图随机切分、两套 YOLO 数据转换
  train.py               顺序训练 pose 与 ear 模型，并各自测试
  infer_labelme.py       两模型推理、同株配对、LabelMe 回写
  evaluate.py            点位像素误差及两个框的 IoU
  pose_pipeline/core.py  标签、ROI、格式规则
  tests/                 无 GPU 的转换与合并测试
```

仓库保留 `models/` 中的两套训练后权重及推理元数据。`data/`、`runs/`、原始图片、标注、推理 JSON 和运行日志均不提交；训练数据需单独保存。

## 2. 本地转换并打包训练数据

以下命令在本地 PowerShell、从 `D:\project\课题组\yolo11_pose_pipeline` 运行：

```powershell
cd 'D:\project\课题组\yolo11_pose_pipeline'
python -m pytest -q
python build_dataset.py --source '..\玉米数据0918\iPhone17' --audit-only
python build_dataset.py --source '..\玉米数据0918\iPhone17' --output 'data\image_split\pose' --ear-output 'data\image_split\ear'
tar -czf 'data\training_bundle_image_split.tar.gz' -C 'data\image_split' pose ear
```

`--audit-only` 只是程序检查重复标签、类型、坐标、group_id、裁图覆盖情况并打印数据统计；它不改文件，也不要求再人工逐张审阅。你修正后三张图片后，当前 171 张审计的 `warnings` 已为空。若缺少本地依赖，可运行 `pip install Pillow PyYAML numpy pytest`。`build_dataset.py` 在本地直接生成训练用裁图和标签；两个输出目录必须为空。归档仅包含训练所需的 `data/pose` 与 `data/ear`，不上传原始 LabelMe JSON 及其重复嵌入的 `imageData`。

旧的 `data/training_bundle.tar.gz` 使用按编号分组的切分策略。可以按上面命令从原始标注重建，也可以在服务器解压旧包后运行 `resplit_dataset.py`，直接对已生成的裁图和 YOLO 标签重新划分；不需要改动原始标注。不要把旧归档直接当作逐图划分的数据训练。

如果 PowerShell 环境没有 `tar`，也可用 `Compress-Archive -Path 'data\image_split\pose','data\image_split\ear' -DestinationPath 'data\training_bundle_image_split.zip'`，远端改用 `unzip` 解压。可检查 `data/image_split/pose/splits.csv` 确认逐图划分结果；默认不按植株身份约束切分。

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

建议先用约 24 GB 显存的 NVIDIA GPU，`imgsz=1024, batch=4` 试跑。两套模型顺序训练，不需要同时驻留显存。16 GB 可先试 batch 2。请准备 Linux、Python 3.10–3.12、可用的 NVIDIA 驱动以及足够的数据与检查点磁盘空间。

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
scp 'D:\project\课题组\yolo11_pose_pipeline\data\training_bundle_image_split.tar.gz' USER@HOST:~/maize-pose/data/
```

然后在远端仓库运行：

```bash
cd ~/maize-pose
source .venv/bin/activate
tar -xzf data/training_bundle_image_split.tar.gz -C data
test -f data/pose/data.yaml
test -f data/ear/data.yaml
```

如果上传的是旧的 `training_bundle.tar.gz`，先解压到临时目录，再用新脚本生成 `data/pose` 和 `data/ear`：

```bash
mkdir -p data/imported
tar -xzf data/training_bundle.tar.gz -C data/imported
python resplit_dataset.py --source data/imported --output data
```

新输出目录必须为空；脚本会保持 pose 和 ear 的图片划分一致，并把结果写入 `data/pose/splits.csv`。

解压后目录应包含：

- `data/pose`：裁图、41 列 pose 标签、`data.yaml`、`metadata.json`、`splits.csv`。
- `data/ear`：同样的裁图、5 列 ear 检测标签、`data.yaml`。程序先尝试硬链接 pose 裁图以节省空间，文件系统不支持时复制。

本地转换时默认逐张图片随机切分，固定随机种子 42；171 张图片按默认 70%/15%/15% 比例约为 train 119、val 26、test 26 张。相邻照片可以进入不同集合，因此源设备上的 val/test 分数可能偏乐观；第 7 节的独立 PM 样本用于检验跨设备效果。LabelMe 的 `group_id` 只负责关联单张图内的框和点，不参与数据切分。若之后需要恢复按编号分块，可传 `--block-size 20`；若有明确的拍摄段分组，可建立覆盖全部图片的 `image,group` CSV，再用 `--groups-csv` 构建到新的空目录。示例：

```csv
image,group
IMG_2866.jpeg,plant-a
IMG_2867.jpeg,plant-a
```

## 5. 训练与源设备测试

```bash
python train.py --pose-data data/pose/data.yaml --ear-data data/ear/data.yaml --task both --imgsz 1024 --batch 4 --epochs 150 --patience 30 --device 0
```

默认顺序训练 `yolo11s-pose.pt` 和 `yolo11s.pt`，分别用源设备 test 划分评估。若只需重训其中一个模型，使用 `--task pose` 或 `--task ear`。终端会输出两套实际 `best.pt` 路径；同名实验重跑时 Ultralytics 可能给目录追加编号。显存不足时先减 `--batch`，再考虑降低 `--imgsz`。

训练入口会在服务器上自动生成 `data.runtime.yaml`，把数据根路径改为解压后的实际绝对路径。因此本地 `data.yaml` 中的 Windows 路径不会影响远端训练；不需要在服务器重新运行 `build_dataset.py`。

调参先看 val；test 只用于最终检验。由于源图数量少，还需要以下 PM 独立评估。

## 6. iPhone17PM 推理与复核

把待推理 JPEG 放入 `data/pm/images/`，运行：

```bash
python infer_labelme.py --pose-weights models/pose-best.pt --ear-weights models/ear-best.pt --source data/pm/images --metadata models/metadata.json --imgsz 1024 --device 0
```

成功预测的图像旁生成同名 JSON，含 12 点及 `plant`、`ear` 两个框。已有 JSON 默认跳过；明确要替换时才传 `--overwrite`。这些模型在部分 iPhone17PM 图片上产生过明显错误的点位；生成 JSON 不代表标注正确，使用前需人工复核。

程序打印低置信度点数、雌穗框是否由关键点估计，以及没有检出中央植株的图名。没有植株检测时不写伪造 JSON，需人工处理或重新推理。下载结果时将 JSON 与同名 JPEG 放在同一目录。LabelMe 中如框遮住点，可先在形状列表里隐藏框，调整点后再显示框。

## 7. 跨设备评估与下一轮训练

在首次训练前从 PM 中挑 20–30 张覆盖遮挡、密集邻株、不同叶位的图，**人工复核**后另存为 `data/pm_gold/*.json`，不参与首次训练。对相同图片推理后：

```bash
python evaluate.py --gold data/pm_gold --pred data/pm/images
```

评估输出 12 点在原图上的误差中位数、P90、25/50/100 像素内比例，另给 `plant` 与 `ear` 框的 IoU 中位数和漏预测图名。再记录每张图人工复核耗时，与从零标注比较。若耳框差，可单独改进 ear 检测模型；若关键点差，可提高 pose 的 `imgsz`、检查裁图及增加人工复核数据。未经复核的 AI 预测不应直接混入训练集。

## 8. 当前验证范围

已对真实 171 张运行只读审计，格式转换和合并 JSON 的测试通过。pose 与 ear 模型已分别完成训练，选出的权重保存在 `models/`。iPhone17PM 的批量推理暴露了跨设备泛化问题：部分图片虽然生成完整 JSON，关键点位置仍明显错误。需要人工复核的 PM 样本评估和改进模型，不能直接把批量预测当作可靠标注。
