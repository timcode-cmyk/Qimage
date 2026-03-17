import os
import argparse
from pathlib import Path
from rembg import remove, new_session
from PIL import Image, ImageChops, ImageFilter

def process_images(input_dir, output_dir, bg_color='transparent', model_name='isnet-general-use', refine=False):
    """
    遍历文件夹处理图片：去背并可选合成背景。
    
    Args:
        input_dir (str): 输入文件夹路径
        output_dir (str): 输出文件夹路径
        bg_color (str): 'transparent', 'green', 'blue' 或十六进制颜色代码
        model_name (str): 使用的 rembg 模型名称
        refine (bool): 是否启用二次修复模式
    """
    
    # 定义颜色映射
    colors = {
        'green': (0, 255, 0),   # 绿幕
        'blue': (0, 0, 255),    # 蓝幕
        'white': (255, 255, 255),
        'black': (0, 0, 0)
    }

    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # 如果输出目录不存在，则创建
    if not output_path.exists():
        output_path.mkdir(parents=True)
        print(f"已创建输出目录: {output_path}")

    # 支持的图片格式
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    
    # 获取所有图片文件
    files = [f for f in input_path.iterdir() if f.suffix.lower() in valid_extensions]
    total_files = len(files)
    
    print(f"开始处理，共发现 {total_files} 张图片...")
    print(f"模式: 去除背景 + {bg_color} 背景")
    print(f"使用模型: {model_name}")
    if refine:
        print("修复模式: 已激活 (使用 u2net_human_seg 进行二次校验)")

    # 初始化 rembg 会话 (使用 onnxruntime)
    print(f"正在加载 {model_name} 模型 (首次运行会自动下载)...")
    session = new_session(model_name)
    refine_session = None
    if refine:
        refiner_model_name = "u2net_human_seg"
        print(f"正在加载修复模型 {refiner_model_name}...")
        refine_session = new_session(refiner_model_name)

    for index, file_path in enumerate(files, 1):
        try:
            print(f"[{index}/{total_files}] 正在处理: {file_path.name} ...", end="", flush=True)
            
            # 1. 打开原图
            img = Image.open(file_path).convert('RGB')
            
            # 2. 使用 rembg 去除背景 (传入 session 避免重复加载模型)
            result = remove(img, session=session)

            # 如果启用了修复模式，则进行二次处理
            if refine and refine_session:
                # 使用修复模型处理原图
                refine_result = remove(img, session=refine_session)

                # 提取主模型和修复模型的 Alpha 通道 (蒙版)
                main_alpha = result.getchannel('A')
                refine_alpha = refine_result.getchannel('A')

                # 关键改进：对修复蒙版进行"收缩" (腐蚀) 和羽化
                # 原因：修复模型(u2net)边缘较粗糙，直接叠加会导致发丝细节丢失变成一坨。
                # 解决：先将修复蒙版向内收缩 (MinFilter)，只保留身体内部的核心区域，
                # 然后轻微模糊 (GaussianBlur) 使其自然融合，避免覆盖主模型的精细边缘。
                # MinFilter(9) 约等于收缩4像素半径，足以避开大部分发丝边缘
                refine_alpha = refine_alpha.filter(ImageFilter.MinFilter(9))
                refine_alpha = refine_alpha.filter(ImageFilter.GaussianBlur(radius=1))

                # 合并两个 Alpha 通道：取两个蒙版中更不透明的像素
                # 这相当于取了两个模型识别出的前景区域的"并集"
                combined_alpha = ImageChops.lighter(main_alpha, refine_alpha)

                # 将合并后的 Alpha 通道应用回原图
                result = img.copy()
                result.putalpha(combined_alpha)

            # 3. 处理背景合成
            if bg_color != 'transparent':
                # 获取目标背景色
                color_rgb = colors.get(bg_color, colors['green']) # 默认绿色
                
                # 创建纯色背景图 (RGB模式)
                background = Image.new("RGB", result.size, color_rgb)
                
                # 将去背后的图层(result) 粘贴到 纯色背景上
                # 第三个参数 result 是 mask，用于处理半透明边缘
                background.paste(result, (0, 0), result)
                final_image = background
                
                # 有背景通常保存为 JPG 以减小体积，也可以是 PNG
                output_filename = f"{file_path.stem}_no_bg_{bg_color}.jpg"
                save_format = "JPEG"
            else:
                # 保持透明，直接使用 result
                final_image = result
                # 透明背景必须保存为 PNG
                output_filename = f"{file_path.stem}_no_bg.png"
                save_format = "PNG"

            # 4. 保存文件
            save_path = output_path / output_filename
            final_image.save(save_path, format=save_format)
            
            print(" 完成")

        except Exception as e:
            print(f" 失败! 错误: {e}")

    print("-" * 30)
    print(f"处理完成！所有图片已保存至: {output_path.absolute()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量图片去背与绿幕/蓝幕合成工具")
    
    parser.add_argument("-i", "--input", required=True, help="输入图片文件夹路径")
    parser.add_argument("-o", "--output", required=True, help="输出保存文件夹路径")
    parser.add_argument("-c", "--color", default="transparent", choices=['transparent', 'green', 'blue', 'white', 'black'], 
                        help="背景颜色选项: transparent(默认), green(绿幕), blue(蓝幕), white, black")
    parser.add_argument("-m", "--model", default="isnet-general-use", 
                        choices=['u2net', 'u2netp', 'u2net_human_seg', 'u2net_cloth_seg', 'silueta', 'isnet-general-use', 'isnet-anime', 'sam'],
                        help="选择模型: isnet-general-use(默认高精度), u2net(通用), isnet-anime(动漫), u2net_human_seg(人像)等")
    parser.add_argument("--refine", action="store_true", 
                        help="使用 u2net_human_seg 模型进行二次修复，防止主体被误删 (主要针对人像)")

    args = parser.parse_args()

    process_images(args.input, args.output, args.color, args.model, args.refine)
