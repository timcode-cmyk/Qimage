"""
基于 Intel MiDaS / DPT 的深度估计模块。
纯 ONNX Runtime 推理，无 torch/transformers，便于打包与部署。
模型来自 Intel（海外团队），ONNX 格式托管于 Hugging Face。
输出灰度深度图。
"""
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

# MiDaS v2.1 Small (Intel)，约 66MB，输入 256x256
MODEL_REPO = "julienkay/sentis-MiDaS"
MODEL_FILE = "onnx/midas_v21_small_256.onnx"
INPUT_SIZE = 256
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _get_model_path() -> Path:
    """通过 huggingface_hub 下载 MiDaS ONNX 模型。"""
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename=MODEL_FILE,
            local_dir=None,
        )
        return Path(path)
    except Exception as e:
        # 回退：使用直接 URL 下载
        import urllib.request
        cache_dir = Path.home() / ".cache" / "qimage" / "models"
        cache_dir.mkdir(parents=True, exist_ok=True)
        model_path = cache_dir / "midas_v21_small_256.onnx"
        if not model_path.exists():
            url = f"https://huggingface.co/{MODEL_REPO}/resolve/main/{MODEL_FILE}"
            print(f"正在下载深度模型 (~66MB)...")
            urllib.request.urlretrieve(url, model_path)
        return model_path


def _get_onnx_providers() -> list[str]:
    """根据平台选择最佳 ONNX 执行提供者。macOS 用 CoreML 加速，NVIDIA 用 CUDA。"""
    available = ort.get_available_providers()
    providers = []
    if "CoreMLExecutionProvider" in available:
        providers.append("CoreMLExecutionProvider")  # macOS GPU/ANE 加速
    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")   # NVIDIA GPU
    providers.append("CPUExecutionProvider")
    return providers


class DepthEstimator:
    """Intel MiDaS 深度估计器，纯 ONNX Runtime，输出灰度深度图。"""

    def __init__(self, model_path: str | Path | None = None):
        if model_path is None:
            model_path = _get_model_path()
        self.model_path = Path(model_path)
        providers = _get_onnx_providers()
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=ort.SessionOptions(),
            providers=providers,
        )
        self.input_name = self.session.get_inputs()[0].name

    def _preprocess(self, image: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        orig_h, orig_w = image.shape[:2]
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = image.astype(np.float32) / 255.0
        image = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        image = (image - MEAN) / STD
        image = image.transpose(2, 0, 1)[None].astype(np.float32)
        return image, (orig_h, orig_w)

    def infer(self, image: np.ndarray) -> np.ndarray:
        """
        对单张图像进行深度估计，返回灰度深度图 (0-255)。

        Args:
            image: BGR 或 RGB 图像 (numpy array, H x W x C)

        Returns:
            灰度深度图 (H x W, uint8)，近处亮、远处暗
        """
        input_tensor, (orig_h, orig_w) = self._preprocess(image)
        depth = self.session.run(None, {self.input_name: input_tensor})[0]

        if depth.ndim == 4:
            depth = depth[0, 0]
        else:
            depth = depth[0]

        depth_min, depth_max = depth.min(), depth.max()
        if depth_max - depth_min > 1e-6:
            depth = (depth - depth_min) / (depth_max - depth_min) * 255.0
        else:
            depth = np.zeros_like(depth)
        depth = depth.astype(np.uint8)
        depth = cv2.resize(depth, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        return depth


def estimate_depth(image: np.ndarray, model_path: str | Path | None = None) -> np.ndarray:
    """便捷函数：估计图像深度并返回灰度深度图。"""
    estimator = DepthEstimator(model_path)
    return estimator.infer(image)
