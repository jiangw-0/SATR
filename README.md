# SATR 代码说明（中文 README）

本仓库包含我们项目 **Support-Aware Transmittance Refinement (SATR)** 的主要代码，用于图像生成中的细粒度遮挡控制。

本次代码发布 **不包含数据集**。不过，只要用户按照项目约定准备 **自定义 scene JSON 文件**，仍然可以运行代码。

---

## 1. 代码中包含什么？

整个代码库大致可以分为三部分：

1. **我们的方法（推荐主入口）**
   - 两阶段生成流程：Stage I 支持区域发现 + Stage II 支持引导 refine。
   - 使用 LaRender 风格的 latent rendering、attention-guided SAM 掩码提取，以及 refine 阶段的 support reinjection。

2. **基线 / 对比方法运行脚本**
   - 论文中的 LaRender baseline。
   - Prompt-driven 的 SDXL / SD 1.5 / SD-Turbo / SDXS / SSD-1B 等运行脚本。
   - GLIGEN baseline 运行脚本。

3. **评测与工具脚本**
   - 我们的 bbox 格式与 paper baseline bbox 格式之间的 JSON 转换。
   - 结构感知评测脚本。
   - 后处理、结果重选、子集整理等辅助脚本。

---

## 2. 哪些文件是真正有用的？

### 2.1 我们方法的核心文件

如果你只想保留 SATR 主流程，最重要的是下面这些文件：

- `demos_occlusion_sam_attn.py`
  - 我们方法的主入口。
  - 执行 **Stage I draft 生成 + 基于 SAM 的支持区域估计 + Stage II refine + best-of-K 选择**。

- `larender_sdxl_sam_attn.py`
  - 我们方法所使用的、基于 LaRender 修改后的核心实现。
  - 负责 latent rendering、transmittance 建模、attention 导出、density scheduling、mask/attention 融合等。

- `sam_masker_v6.py`
  - attention-guided SAM 掩码提取模块。
  - 使用正负提示点与后处理，提取每个对象的 mask。

- `utils.py`
  - 通用工具函数。
  - 包括 prompt 截断、token index 辅助、文件系统处理、seed 设置等。

- `run_dataset_json_dir.sh`
  - 推荐的批量运行入口。
  - 用于对一个目录中的 scene JSON 文件批量运行我们的方法。

### 2.2 论文 LaRender baseline 的核心文件

- `demos_occlusion_batch.py`
  - 原始 / 论文风格 LaRender baseline 的主运行脚本。
  - 输入要求是 paper 格式的 bbox。

- `larender_sdxl.py`
  - baseline 所使用的原始 LaRender 风格 SDXL 实现。

- `run_paper_json_dir.sh`
  - 对一个目录中的 JSON scene 批量运行 paper baseline。

- `convert_json_bbox_to_paper.py`
  - 用于把我们方法使用的 bbox 格式 `[y0,y1,x0,x1]` 转成 paper baseline 使用的 `[x0,x1,y0,y1]`。

### 2.3 评测与分析文件

- `eval_occlusion_v24_structv3.py`
  - 当前发布版本中推荐使用的评测脚本。
  - 计算我们项目中的结构感知评测指标。

- `run_compare_dataset.sh`
  - 一个完整的对比流程脚本：
    1. 把 JSON 转成 paper baseline 格式；
    2. 跑我们的方法；
    3. 跑 paper baseline；
    4. 对两者进行评测。

- `reselect_refine_best_from_eval.py`
  - 根据评测结果重新选择 best refine 图像。


## 3. 代码默认的目录结构

代码默认假设项目根目录结构大致如下：

```text
SATR/
├── src/
│   ├── demos_occlusion_sam_attn.py
│   ├── larender_sdxl_sam_attn.py
│   ├── sam_masker_v6.py
│   ├── utils.py
│   ├── run_dataset_json_dir.sh
│   ├── demos_occlusion_batch.py
│   ├── larender_sdxl.py
│   ├── run_paper_json_dir.sh
│   ├── convert_json_bbox_to_paper.py
│   ├── eval_occlusion_v24_structv3.py
│   └── run_compare_dataset.sh
├── models/
│   ├── sam/
│   │   └── sam_vit_h_4b8939.pth
│   └── ... （扩散模型权重 / Hugging Face cache）
├── outputs/
├── json/          # 我们格式的 scene JSON
└── json_paper/    # 转成 paper baseline 格式后的 JSON
```

建议设置项目根目录环境变量：

```bash
export OCCLUSION_ROOT=/path/to/SATR
```

很多脚本会通过 `OCCLUSION_ROOT` 去寻找 `src/`、`models/`、`outputs/` 等目录。

---

## 4. 依赖环境

推荐环境：**Python 3.10+**，并使用支持 CUDA 的 PyTorch。

主要依赖安装方式：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install diffusers transformers accelerate safetensors
pip install opencv-python pillow numpy pandas
pip install segment-anything
pip install spacy
python -m spacy download en_core_web_sm
```

如果希望减小显存占用，可选安装：

```bash
pip install xformers
```

说明：
- `segment-anything` 在 Stage I mask 提取和评测中都会用到。
- `spacy` 在 `utils.py` 中会用到，建议安装 `en_core_web_sm`。
- 评测脚本中还可能会用到来自 Hugging Face 的 CLIP / DPT 模型。

---

## 5. 所需模型权重与环境变量

### 5.1 我们的方法

代码通常会尝试加载：

- **IterComp**（默认预训练 backbone）
- 备用：**SDXL base**
- **SAM checkpoint**

建议设置如下环境变量：

```bash
export OCCLUSION_ROOT=/path/to/SATR
export ITERCOMP_MODEL=/path/to/local/IterComp
export SDXL_BASE_MODEL=/path/to/local/sdxl-base-1.0
export SAM_CHECKPOINT=/path/to/SATR/models/sam/sam_vit_h_4b8939.pth
export SAM_MODEL_TYPE=vit_h
export HF_HUB_CACHE=/path/to/hf_cache
```

如果需要离线运行：

```bash
export LOCAL_FILES_ONLY=1
export HF_HUB_OFFLINE=1
```

### 5.2 GLIGEN baseline（可选）

```bash
export GLIGEN_PATH=/path/to/gligen-xl-1024
```

---

## 6. Scene JSON 输入格式（自定义输入）

由于数据集 **不公开**，因此使用者需要自行准备 scene JSON 文件。

### 6.1 我们方法要求的 bbox 格式

```text
[y0, y1, x0, x1]
```

### 6.2 最小示例（`json/scene_template_ours.json`）

```json
{
  "name": "scene_template_ours",
  "pretrain": "IterComp",
  "height": 1024,
  "width": 1024,
  "seed": 123,
  "num_draft": 6,
  "num_refine": 16,
  "negative_prompt": "cartoon, illustration, CGI, lowres, blurry",
  "density_scheduler": "inverse_proportional",
  "mask_fusion": {
    "mode": "sam_attn_blend",
    "lambda_start": 0.72,
    "lambda_end": 0.62,
    "attn_power": 1.25,
    "attn_floor": 0.18
  },
  "objects": [
    {
      "name": "background",
      "prompt": "a greenhouse background with plants, realistic photo",
      "bbox": [0.0, 1.0, 0.0, 1.0],
      "alpha": 0.999,
      "use_sam": false,
      "use_attn": false
    },
    {
      "name": "statue",
      "prompt": "a marble statue bust, realistic photo <<behind:web>>",
      "bbox": [0.28, 0.72, 0.22, 0.86],
      "alpha": 0.95,
      "use_sam": true,
      "use_attn": true,
      "keywords": ["statue"]
    },
    {
      "name": "web",
      "prompt": "a spider web in the foreground, thin strands, visible holes, realistic photo <<front_of:statue>>",
      "bbox": [0.24, 0.76, 0.24, 0.80],
      "alpha": 0.80,
      "use_sam": true,
      "use_attn": true,
      "keywords": ["web"]
    }
  ],
  "occlusion_pairs": [
    {
      "back": 1,
      "front": 2,
      "mode": "holey",
      "min_occl": 0.08,
      "max_occl": 0.22,
      "back_density_min": 0.14,
      "front_occupancy_min": 0.12,
      "intersection_max": 0.18
    }
  ]
}
```

### 6.3 说明
- `objects` 的顺序是 **从后到前**。
- `objects[0]` 一般是 background。
- 类似 `<<behind:web>>`、`<<front_of:statue>>` 这样的关系指令会被方法解析并转成遮挡约束。
- `keywords` 用于 attention-guided support extraction 时的 token 选择。

---

## 7. 如何运行我们的方法

### 7.1 批量运行一个 JSON 目录（推荐）

将你自己的 JSON 文件放到：

```text
$OCCLUSION_ROOT/json/
```

然后运行：

```bash
cd $OCCLUSION_ROOT/src
bash run_dataset_json_dir.sh $OCCLUSION_ROOT/json $OCCLUSION_ROOT/outputs/ours
```

如果你想显式指定分辨率：

```bash
bash run_dataset_json_dir.sh $OCCLUSION_ROOT/json 1024 1024 $OCCLUSION_ROOT/outputs/ours
```

可选环境变量：

```bash
export OCCLUSION_K=16
export FORCE_RERUN=1
```

### 7.2 直接运行单个 scene

```bash
python demos_occlusion_sam_attn.py \
  --scene $OCCLUSION_ROOT/json/example_scene.json \
  --out   $OCCLUSION_ROOT/outputs/example_scene
```

### 7.3 我们方法的输出内容

每个 scene 的输出目录通常会包含：

- `draft_*.jpg` —— Stage I draft 候选图
- `refine_*.jpg` —— Stage II refine 候选图
- `refine_best.jpg` —— 最终选出的结果
- `scene_run_summary.json` —— 运行摘要
- 可选的 debug masks / attention maps / support 可视化结果

---

## 8. 如何运行论文中的 LaRender baseline

paper baseline 使用的 bbox 格式是：

```text
[x0, x1, y0, y1]
```

所以如果你的 scene JSON 是我们方法的格式，需要先做转换。

### 8.1 先转换 scene JSON

```bash
python convert_json_bbox_to_paper.py \
  --in_dir  $OCCLUSION_ROOT/json \
  --out_dir $OCCLUSION_ROOT/json_paper \
  --overwrite \
  --reorder_by_pairs
```

### 8.2 运行 paper baseline

```bash
bash run_paper_json_dir.sh \
  $OCCLUSION_ROOT/json_paper \
  1024 1024 \
  $OCCLUSION_ROOT/outputs/paper \
  16
```

### 8.3 GLIGEN 变体（可选）

```bash
python run_larender_gligen_batch.py \
  --scenes_dir $OCCLUSION_ROOT/json_paper \
  --out_root   $OCCLUSION_ROOT/outputs/paper_gligen \
  --model_root $GLIGEN_PATH \
  --num_images 6
```

---

## 9. 如何运行评测

当前发布版本推荐的评测脚本是：

- `eval_occlusion_v24_structv3.py`

### 9.1 评测我们方法的输出

```bash
python eval_occlusion_v24_structv3.py \
  --images_root $OCCLUSION_ROOT/outputs/ours \
  --scenes_dir  $OCCLUSION_ROOT/json \
  --sam_ckpt    $SAM_CHECKPOINT \
  --bbox_format y1y2x1x2 \
  --image_glob  "refine_*.jpg" \
  --scalar_mode scalar50 \
  --out_csv     ours_eval.csv \
  --out_json    ours_eval.json
```

### 9.2 评测 paper baseline 的输出

```bash
python eval_occlusion_v24_structv3.py \
  --images_root $OCCLUSION_ROOT/outputs/paper \
  --scenes_dir  $OCCLUSION_ROOT/json_paper \
  --sam_ckpt    $SAM_CHECKPOINT \
  --bbox_format x0x1y0y1 \
  --image_glob  "*.png" \
  --scalar_mode scalar50 \
  --out_csv     paper_eval.csv \
  --out_json    paper_eval.json
```

### 9.3 可选的 CLIP / 深度模型

```bash
python eval_occlusion_v24_structv3.py \
  ... \
  --clip_model openai/clip-vit-large-patch14 \
  --depth_model Intel/dpt-hybrid-midas
```

---

## 10. 一键运行完整对比流程

如果你已经准备好我们格式的 scene，并希望一次性完成：

1. JSON 转换；
2. 运行我们的方法；
3. 运行 paper baseline；
4. 对两者进行评测；

可以直接运行：

```bash
bash run_compare_dataset.sh $SAM_CHECKPOINT
```

运行前建议设置：

```bash
export OCCLUSION_ROOT=/path/to/SATR
export SAM_CHECKPOINT=/path/to/SATR/models/sam/sam_vit_h_4b8939.pth
```

可选覆盖变量：

```bash
export JSON_DIR=$OCCLUSION_ROOT/json
export JSON_PAPER_DIR=$OCCLUSION_ROOT/json_paper
export OUT_OURS=$OCCLUSION_ROOT/outputs/ours_ds
export OUT_PAPER=$OCCLUSION_ROOT/outputs/paper_ds
export EVAL_OUT=$OCCLUSION_ROOT/outputs/eval_ds
export OCCLUSION_K=16
export PAPER_NUM_IMAGES=16
```

---


## 12. 复现说明 / 常见问题

### 12.1 找不到 SAM checkpoint

请设置：

```bash
export SAM_CHECKPOINT=/path/to/sam_vit_h_4b8939.pth
```

### 12.2 离线环境下 IterComp / SDXL 无法加载

请显式设置本地模型路径：

```bash
export ITERCOMP_MODEL=/path/to/local/IterComp
export SDXL_BASE_MODEL=/path/to/local/sdxl-base-1.0
```

同时建议设置：

```bash
export LOCAL_FILES_ONLY=1
export HF_HUB_OFFLINE=1
```

### 12.3 bbox 格式不一致导致结果异常

请务必注意：

- **我们的方法** 使用 `[y0, y1, x0, x1]`
- **paper baseline** 使用 `[x0, x1, y0, y1]`

如果需要跑 paper baseline，请先使用 `convert_json_bbox_to_paper.py` 转换。

### 12.4 prompt 关系没有生效

请确认：
- `objects` 顺序是否为 **从后到前**；
- `occlusion_pairs` 是否和对象顺序一致；
- 关系指令（如 `<<behind:web>>`）是否与对象名一致。

---

