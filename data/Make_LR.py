import os
import cv2
from PIL import Image
if __name__ == '__main__':
    # 获取路径
    py_path = os.path.abspath(__file__) 
    file_dir = os.path.dirname(py_path) 

    # 数据文件夹
    dataset_folder = os.path.join(os.path.dirname(os.path.dirname(file_dir)), 'PanDataset')
    data_folder = os.path.join(os.path.join(dataset_folder), 'GF2_data', 'test128')
    ms_HR_folder = os.path.join(os.path.join(data_folder), 'ms_HR')
    pan_folder = os.path.join(os.path.join(data_folder), 'pan')
    # 读取图像文件列表
    ms_HR_files = [f for f in os.listdir(ms_HR_folder) if f.endswith('.tif') ]
    # 设置降采样因子（Wald协议）,降采样因子设为4
    downsample_factor = 4  

    # 创建低分辨率图像的存储路径
    ms_LR_folder = os.path.join(os.path.join(data_folder), 'ms_LR')
    os.makedirs(ms_LR_folder, exist_ok=True)
    
    # 降采样过程
    def downsample_image(img, factor):
        height, width = img.shape[:2]
        new_dim = (width // factor, height // factor)
        downsampled_img = cv2.resize(img, new_dim, interpolation=cv2.INTER_LINEAR)  # 双线性插值
        return downsampled_img

    # 对所有图像进行降采样
    for image_file in ms_HR_files:
        ms_path = os.path.join(ms_HR_folder, image_file)
        img = cv2.imread(ms_path)
        if img is None:
            print(f"Unable to open image {image_file}, deleting it.")
            os.remove(ms_path)
            pan_path = os.path.join(pan_folder, image_file)
            os.remove(pan_path)
            continue
        try:
            # 使用PIL转换BGR到RGB
            img_pil = Image.open(ms_path).convert('RGB')
        except Exception as e:
            print(f"Error converting {image_file} to RGB: {e}, deleting it.")
            os.remove(ms_path)
            pan_path = os.path.join(pan_folder, image_file)
            if os.path.exists(pan_path):
                os.remove(pan_path)
            continue
        
        lr_image = downsample_image(img, downsample_factor)
        # 保存降采样后的低分辨率图像
        lr_image_path = os.path.join(ms_LR_folder, image_file)
        cv2.imwrite(lr_image_path, lr_image)
        
