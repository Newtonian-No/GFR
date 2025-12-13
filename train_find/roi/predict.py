# TransUNet对单张图片进行分割
import argparse
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset
#from back.networks.vit_seg_modeling_mamba import VisionTransformer as ViT_seg
from back.networks.vit_seg_modeling_bimambaattention import VisionTransformer as ViT_seg
from back.networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
import pydicom
from scipy.ndimage import zoom
import SimpleITK as sitk

class SingleImageDataset(Dataset):
    def __init__(self, image_path, transform=None):
        self.image_path = image_path
        self.transform = transform
        print(image_path)

    # 返回数据集的长度
    def __len__(self):
        return 1

    def normalize_image_data(self, image_data):
        dicom_array = image_data.pixel_array.astype(np.float32)
        # 将值的范围截断在0到最大值的一半之间
        max_value = np.max(dicom_array)
        dicom_array = np.clip(dicom_array, 0, max_value / 2)
        # 归一化到 [0, 1] 范围
        dicom_array /= (max_value / 2)
        return dicom_array

    # 获取数据集中的一个元素
    def __getitem__(self, idx):
        image_data = pydicom.dcmread(self.image_path, force=True)
        image = self.normalize_image_data(image_data)
        sample = {'image': image}
        if self.transform:
            sample = self.transform(sample)
        # 分离文件名和扩展名
        image_name, image_extension = os.path.splitext(os.path.basename(self.image_path))
        sample['case_name'] = image_name
        print(image_name)
        return sample
    

class RandomGenerator(object): # 用于测试时的图像大小调整, 不进行数据增强和标签处理
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image= sample['image']

        # 调整图像和标签的大小
        if image.ndim == 2:
            x, y = image.shape
        elif image.ndim == 3:
            x, y, _ = image.shape
        else:
            raise ValueError(f"Unexpected number of dimensions in the image: {image.ndim}")

        if x != self.output_size[0] or y != self.output_size[1]:
            if image.ndim == 2:
                image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            elif image.ndim == 3:
                image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y, 1), order=3)
                image = np.squeeze(image)

        # 转换为PyTorch张量
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(-3)
        sample = {'image': image}
        return sample

#####用于独立测试集#####
def test_single_volume1(image,  net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1, device=None):
    image = image.squeeze(0)
    image = image.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(image)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().to(device)
            net.eval()
            with torch.no_grad():
                outputs = net(input)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().to(device)
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    

    if test_save_path is not None:
        
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))

        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))

        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")

        sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
    
    return prediction


def segment_kidney(image_path, model_path, img_size=224, num_classes=3):
    print(f"步骤3: 对{image_path}进行肾脏分割...")

    # 设置随机种子
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1234)
    cudnn.benchmark = False
    cudnn.deterministic = True
    
    # 加载模型配置
    vit_name = 'R50-ViT-L_16'
    
    vit_patches_size = 16
    config_vit = CONFIGS_ViT_seg[vit_name]
    config_vit.n_classes = num_classes
    config_vit.patches.size = (vit_patches_size, vit_patches_size)
    if vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(img_size/vit_patches_size), int(img_size/vit_patches_size))

    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = ViT_seg(config_vit, img_size=img_size, num_classes=config_vit.n_classes)
    net = net.to(device)
    
    # 加载模型权重
    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_path))
    else:
        net.load_state_dict(torch.load(model_path, map_location=device))

    # if device == 'cuda':
    #     net.load_state_dict(torch.load(model_path))
    # else:
    #     net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    print("模型加载成功.")

    # 预测
    output_dir = "output/segmentation"
    net.eval()
    dataset = SingleImageDataset(image_path, transform=RandomGenerator(output_size=[img_size, img_size]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    
    for sampled_batch in dataloader:
        image, case_name = sampled_batch["image"], sampled_batch['case_name'][0]
        prediction = test_single_volume1(image, net, classes=num_classes, 
                                      patch_size=[img_size, img_size],
                                      test_save_path=output_dir, case=case_name, z_spacing=1, device=device)
        result_path = os.path.join(output_dir, f"{case_name}_pred.nii.gz")
        segmentation = np.squeeze(prediction)
        print("步骤3.5: 调整掩码尺寸以匹配重采样DICOM...")
        # 强制转换为 2D 数组 (H, W)
        if segmentation.ndim == 1:
            try:
                # 假设输入尺寸是 HxW
                segmentation = segmentation.reshape((img_size, img_size))
                print(f"警告: 1D 数组已重塑为 2D: {segmentation.shape}")
            except ValueError:
                print(f"严重错误: 数组维度为 1D 且无法重塑为 {img_size}x{img_size}。跳过调整。")
                # 如果无法重塑，则无法进行缩放，直接使用原始数组
                final_segmentation = segmentation
        if segmentation.ndim >= 2:
            final_segmentation = segmentation.astype(np.int16)
        else:
        # 如果不是 2D 或更高维（比如重塑失败后仍是 1D），则保持原样
            final_segmentation = segmentation

        # 将掩码缩放到与重采样 DICOM 相同的空间尺寸，避免后续叠加错位
        try:
            ds_display = pydicom.dcmread(image_path, force=True)
            target_h, target_w = ds_display.pixel_array.shape
            if final_segmentation.shape != (target_h, target_w):
                scale = (target_h / final_segmentation.shape[0], target_w / final_segmentation.shape[1])
                final_segmentation = zoom(final_segmentation, scale, order=0)
                print(f"掩码已缩放到显示尺寸: {final_segmentation.shape}")

            # 固定的全局方向修正：先水平镜像，再 rot90(k=1)
            final_segmentation = np.fliplr(final_segmentation)
            final_segmentation = np.rot90(final_segmentation, k=1)
            print("已应用固定方向修正：fliplr + rot90(k=1)")

            try:
                prd_itk = sitk.GetImageFromArray(final_segmentation.astype(np.float32))
                spacing = getattr(ds_display, "PixelSpacing", [1.0, 1.0])
                if isinstance(spacing, (list, tuple)) and len(spacing) == 2:
                    prd_itk.SetSpacing((float(spacing[0]), float(spacing[1]), 1.0))
                sitk.WriteImage(prd_itk, result_path)
                print(f"已覆盖保存调整后的掩码: {result_path}")
            except Exception as e:
                print(f"警告: 保存调整后掩码失败，继续使用内存结果。错误: {e}")
        except Exception as e:
            print(f"警告: 缩放掩码到显示尺寸时出错，保持原尺寸。错误: {e}")
        print(f"分割结果路径: {result_path}")
    
        return result_path, final_segmentation
        
        

