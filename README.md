# Qimage

图片/视频去背、深度估计与绿幕合成工具。

## 功能

1. **图片去背** - 去除背景，支持透明或绿幕/蓝幕合成  
2. **深度估计** - 基于 Intel MiDaS ONNX，输出灰度深度图  
3. **视频抠像** - 绿幕合成 或 深度视频（含时序平滑）  
4. **灵活输入** - 支持单个文件或整个文件夹

## 安装

```bash
pip install -r requirements.txt
```

## 使用

### 基本用法

```bash
python main.py -i <输入路径> -o <输出目录> [-m 模式]
```

- `-i, --input`：输入路径（文件或文件夹）  
- `-o, --output`：输出目录  
- `-m, --mode`：`matting` | `depth` | `video`

### 模式说明

| 模式 | 说明 | 示例 |
|------|------|------|
| `matting` | 图片去背，合成背景 | `python main.py -i images/ -o out -m matting -c green` |
| `depth` | 深度估计，输出灰度深度图 | `python main.py -i photo.jpg -o out -m depth` |
| `video` | 绿幕抠像 或 深度视频 | `python main.py -i clip.mp4 -o out -m video [--depth]` |

### 示例

```bash
# 单张图片去背，绿幕
python main.py -i photo.png -o ./out -c green

# 整个文件夹图片去背
python main.py -i ./photos -o ./output -c transparent

# 深度估计（单文件或文件夹）
python main.py -i ./images -o ./depth_out -m depth

# 深度视频（灰度深度图合成为视频，含时序平滑）
python main.py -i video.mp4 -o ./video_out -m video --depth

# 绿幕抠像视频
python main.py -i video.mp4 -o ./video_out -m video

# 人像模式 + 二次修复
python main.py -i portraits/ -o out -m matting -c green --refine --model u2net_human_seg
```

### 参数

| 参数 | 说明 |
|------|------|
| `-c, --color` | 背景颜色：`transparent`, `green`, `blue`, `white`, `black`（仅 matting） |
| `--model` | rembg 模型（matting/video） |
| `--refine` | 人像二次修复（仅 matting） |
| `--depth` | 视频模式下输出深度视频（灰度深度图合成，含时序平滑） |

## 深度模型

深度估计使用 **Intel MiDaS v2.1 Small**（ONNX 格式），仅依赖 `onnxruntime`，**无需 torch/transformers**，适合打包与轻量部署：
- 模型约 66MB
- 纯 ONNX Runtime 推理
- 首次运行自动下载到 Hugging Face 缓存
