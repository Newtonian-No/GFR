import torch
import numpy as np
import cv2
import pydicom
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget
from .vit_seg_modeling_mamba import VisionTransformer, CONFIGS # 导入你的模型

class SegmentationModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        # GradCAM库需要直接获得logits，去除tuple返回
        return self.model(x)

def preprocess_dicom(dicom_path, img_size=224):
    """读取、窗位处理、重采样并归一化 DICOM"""
    ds = pydicom.dcmread(dicom_path)
    img = ds.pixel_array.astype(float)

    # 1. 简单的窗位处理 (以腹部肾脏为例，可根据实际调整)
    # 或者使用 ds.WindowCenter 和 ds.WindowWidth
    wc, ww = 40, 400 
    img_min = wc - ww // 2
    img_max = wc + ww // 2
    img = np.clip(img, img_min, img_max)
    
    # 2. 归一化到 0-1
    img = (img - img_min) / (img_max - img_min)
    
    # 3. 重采样/缩放到模型输入尺寸
    img_resampled = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    
    # 4. 转换为 3 通道 (RGB 格式) 供 Grad-CAM 可视化使用
    img_rgb = np.stack([img_resampled]*3, axis=-1) 
    
    # 5. 转换为模型输入的 Tensor [1, 1, H, W] 
    # 注意：你的模型 forward 会自己处理单通道变三通道
    input_tensor = torch.from_numpy(img_resampled).float().unsqueeze(0).unsqueeze(0)
    
    return input_tensor, img_rgb

def run_grad_cam():
    # 1. 配置模型
    config_name = 'R50-ViT-B_16'  # 根据你的实际配置修改
    config = CONFIGS[config_name]
    config.n_classes = 2      # 修改为你的类别数
    
    config.n_classes = 2 # 假设你的分割任务是 2 类
    model = VisionTransformer(config, img_size=224, num_classes=config.n_classes)
    
    # 加载权重
    model.load_state_dict(torch.load('D:\\Code\\GFR\\back\weights\\best_epoch_weights.pth'))
    model.eval()

    # 预处理 DICOM
    input_tensor, img_rgb = preprocess_dicom('F:\\GFR\\肾动态显像+CT\\HEHONGBAO\\NM\\1.2.840.113619.2.280.2.1.27082024075847014.962359236.dcm')

    # 选择目标层：通常选 Encoder 的最后一个 Norm 层
    target_layers = [model.transformer.encoder.encoder_norm]

    # 初始化 Grad-CAM
    # 由于你的模型返回的是单个 logits，可以直接使用
    cam = GradCAM(model=model, target_layers=target_layers)

    # 定义目标类别 (SemanticSegmentationTarget)
    targets = [SemanticSegmentationTarget(category=1, mask=None)]

    # 生成热力图
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
    
    # 将热力图叠加在重采样后的 DICOM 图像上
    visualization = show_cam_on_image(img_rgb, grayscale_cam, use_rgb=True)
    
    # 保存结果
    cv2.imwrite('dicom_mamba_cam.jpg', visualization)
    print("Grad-CAM 结果已保存为 dicom_mamba_cam.jpg")

if __name__ == '__main__':
    run_grad_cam()