"""
Qimage - 图片/视频去背与处理工具

功能：
- 图片去背、绿幕/蓝幕合成
- 基于 ONNX 的深度估计，输出灰度深度图
- 视频抠像 + 绿幕合成 + 可选深度图
- 支持单文件或文件夹输入
"""
import argparse
from pathlib import Path

from PIL import Image, ImageChops, ImageFilter
from rembg import remove, new_session

# 支持的图片格式
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# 支持的视频格式
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def collect_input_paths(input_path: str | Path) -> list[Path]:
    """
    根据输入路径收集待处理文件。
    支持单文件或文件夹；自动区分图片和视频。
    """
    p = Path(input_path)
    if not p.exists():
        raise FileNotFoundError(f"路径不存在: {p}")

    if p.is_file():
        return [p]
    # 文件夹
    files = []
    for ext in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS:
        files.extend(p.glob(f"*{ext}"))
    return sorted(files)


def process_image(
    file_path: Path,
    output_path: Path,
    bg_color: str = "transparent",
    model_name: str = "isnet-general-use",
    refine: bool = False,
    session=None,
    refine_session=None,
) -> None:
    """处理单张图片：去背并可选合成背景。"""
    colors = {
        "green": (0, 255, 0),
        "blue": (0, 0, 255),
        "white": (255, 255, 255),
        "black": (0, 0, 0),
    }

    img = Image.open(file_path).convert("RGB")
    result = remove(img, session=session)

    if refine and refine_session:
        refine_result = remove(img, session=refine_session)
        main_alpha = result.getchannel("A")
        refine_alpha = refine_result.getchannel("A")
        refine_alpha = refine_alpha.filter(ImageFilter.MinFilter(9))
        refine_alpha = refine_alpha.filter(ImageFilter.GaussianBlur(radius=1))
        combined_alpha = ImageChops.lighter(main_alpha, refine_alpha)
        result = img.copy()
        result.putalpha(combined_alpha)

    if bg_color != "transparent":
        color_rgb = colors.get(bg_color, colors["green"])
        background = Image.new("RGB", result.size, color_rgb)
        background.paste(result, (0, 0), result)
        final_image = background
        output_filename = f"{file_path.stem}_no_bg_{bg_color}.jpg"
        save_format = "JPEG"
    else:
        final_image = result
        output_filename = f"{file_path.stem}_no_bg.png"
        save_format = "PNG"

    save_path = output_path / output_filename
    final_image.save(save_path, format=save_format)


def process_images(
    input_path: str | Path,
    output_dir: str | Path,
    bg_color: str = "transparent",
    model_name: str = "isnet-general-use",
    refine: bool = False,
) -> None:
    """
    遍历输入路径中的图片进行处理：去背并可选合成背景。
    支持单文件或文件夹。
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    files = [f for f in collect_input_paths(input_path) if f.suffix.lower() in IMAGE_EXTENSIONS]
    total = len(files)
    if total == 0:
        print(f"未在 {input_path} 中发现图片文件")
        return

    print(f"开始处理，共 {total} 张图片...")
    print(f"模式: 去除背景 + {bg_color} 背景")
    print(f"使用模型: {model_name}")
    if refine:
        print("修复模式: 已激活 (使用 u2net_human_seg 二次校验)")

    session = new_session(model_name)
    refine_session = new_session("u2net_human_seg") if refine else None

    for i, file_path in enumerate(files, 1):
        try:
            print(f"[{i}/{total}] 正在处理: {file_path.name} ...", end="", flush=True)
            process_image(file_path, output_path, bg_color, model_name, refine, session, refine_session)
            print(" 完成")
        except Exception as e:
            print(f" 失败: {e}")

    print("-" * 30)
    print(f"处理完成，已保存至: {output_path.absolute()}")


def process_depth(
    input_path: str | Path,
    output_dir: str | Path,
) -> None:
    """
    对图片进行深度估计，输出灰度深度图。
    支持单文件或文件夹。
    """
    from depth_dpt import DepthEstimator
    import cv2

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    files = [f for f in collect_input_paths(input_path) if f.suffix.lower() in IMAGE_EXTENSIONS]
    total = len(files)
    if total == 0:
        print(f"未在 {input_path} 中发现图片文件")
        return

    print(f"深度估计模式，共 {total} 张图片...")
    print("正在加载 Intel DPT 深度模型...")
    estimator = DepthEstimator()

    for i, file_path in enumerate(files, 1):
        try:
            print(f"[{i}/{total}] {file_path.name} ...", end="", flush=True)
            img = cv2.imread(str(file_path))
            if img is None:
                raise ValueError("无法读取图像")
            depth = estimator.infer(img)
            out_name = f"{file_path.stem}_depth.png"
            cv2.imwrite(str(output_path / out_name), depth)
            print(" 完成")
        except Exception as e:
            print(f" 失败: {e}")

    print("-" * 30)
    print(f"深度图已保存至: {output_path.absolute()}")


def process_video_matting(
    input_path: str | Path,
    output_dir: str | Path,
    model_name: str = "isnet-general-use",
    with_depth: bool = False,
) -> None:
    """
    视频抠像 + 绿幕合成，可选输出每帧深度图。
    支持单文件或文件夹。
    """
    from video_processor import process_video

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    files = [f for f in collect_input_paths(input_path) if f.suffix.lower() in VIDEO_EXTENSIONS]
    total = len(files)
    if total == 0:
        print(f"未在 {input_path} 中发现视频文件")
        return

    for i, file_path in enumerate(files, 1):
        try:
            out_video = output_path / f"{file_path.stem}_greenscreen.mp4"
            depth_dir = output_path / f"{file_path.stem}_depth" if with_depth else None
            print(f"[{i}/{total}] 处理视频: {file_path.name}")
            process_video(
                file_path,
                out_video,
                model_name=model_name,
                enable_depth=with_depth,
                depth_output_dir=depth_dir,
            )
        except Exception as e:
            print(f"失败: {e}")

    print("-" * 30)
    print(f"视频已保存至: {output_path.absolute()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Qimage - 图片/视频去背、深度估计、绿幕合成工具 (支持文件/文件夹)"
    )
    parser.add_argument("-i", "--input", required=True, help="输入路径（单个文件或文件夹）")
    parser.add_argument("-o", "--output", required=True, help="输出目录")
    parser.add_argument(
        "-m", "--mode",
        choices=["matting", "depth", "video"],
        default="matting",
        help="模式: matting(图片去背), depth(深度图), video(视频抠像绿幕)",
    )
    parser.add_argument(
        "-c", "--color",
        default="transparent",
        choices=["transparent", "green", "blue", "white", "black"],
        help="背景颜色（仅 matting 模式）",
    )
    parser.add_argument(
        "--model",
        default="isnet-general-use",
        choices=["u2net", "u2netp", "u2net_human_seg", "u2net_cloth_seg", "silueta", "isnet-general-use", "isnet-anime", "sam"],
        help="rembg 模型（matting/video）",
    )
    parser.add_argument(
        "--refine",
        action="store_true",
        help="人像二次修复（仅 matting 模式）",
    )
    parser.add_argument(
        "--depth",
        action="store_true",
        dest="video_with_depth",
        help="视频模式下同时输出每帧深度图",
    )
    args = parser.parse_args()

    if args.mode == "matting":
        process_images(args.input, args.output, args.color, args.model, args.refine)
    elif args.mode == "depth":
        process_depth(args.input, args.output)
    elif args.mode == "video":
        process_video_matting(args.input, args.output, args.model, args.video_with_depth)


if __name__ == "__main__":
    main()
