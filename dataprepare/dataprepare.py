import os
import numpy as np
from PIL import Image


def find_all_datasets(data_dir):
    """自动发现所有符合规范的数据集目录"""
    valid_datasets = []
    for dir_name in os.listdir(data_dir):
        dir_path = os.path.join(data_dir, dir_name)
        if os.path.isdir(dir_path):
            if all(os.path.exists(os.path.join(dir_path, sub)) for sub in ['images', 'masks']):
                valid_datasets.append(dir_name)
            else:
                print(f"警告：跳过不完整的数据集目录 {dir_name}")
    return valid_datasets


def process_single_dataset(data_dir, output_dir, dataset_name):
    """处理单个数据集"""
    print(f"\n{'=' * 40}")
    print(f"正在处理数据集: {dataset_name}")

    image_dir = os.path.join(data_dir, dataset_name, 'images')
    mask_dir = os.path.join(data_dir, dataset_name, 'masks')

    # 新增灰度预处理步骤
    preprocess_masks(mask_dir)

    # 获取文件列表并验证
    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])

    # 增强验证逻辑
    if not image_files:
        raise ValueError(f"未找到图片文件：{image_dir}")
    if not mask_files:
        raise ValueError(f"未找到掩码文件：{mask_dir}")
    if len(image_files) != len(mask_files):
        raise ValueError(f"文件数量不匹配：图片{len(image_files)}个 vs 掩码{len(mask_files)}个")

    # 处理图片数据
    image_arrays, mask_arrays = [], []
    for img_file, mask_file in zip(image_files, mask_files):
        # 加载并预处理图片
        img = Image.open(os.path.join(image_dir, img_file)).resize((256, 256)).convert('RGB')
        mask = Image.open(os.path.join(mask_dir, mask_file)).resize((256, 256)).convert('L')

        # 转换格式
        image_arrays.append(np.array(img))
        mask_arrays.append(np.array(mask))

    # 保存数据集
    np.save(os.path.join(output_dir, f'data_{dataset_name}.npy'), np.array(image_arrays))
    np.save(os.path.join(output_dir, f'mask_{dataset_name}.npy'), np.array(mask_arrays))
    print(f"成功保存：{dataset_name} 数据集 ({len(image_files)}个样本)")


def preprocess_masks(mask_folder, overwrite=True):
    """
    预处理掩码文件夹：转换为灰度图并标准化
    参数：
        mask_folder: 掩码目录路径
        overwrite: 是否覆盖原文件
    """
    print(f"开始预处理掩码目录：{mask_folder}")

    for filename in os.listdir(mask_folder):
        file_path = os.path.join(mask_folder, filename)

        try:
            # 只处理图片文件
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            with Image.open(file_path) as img:
                # 转换为灰度图（L模式）
                gray_img = img.convert('L')

                # 标准化尺寸
                resized_img = gray_img.resize((256, 256))

                # 保存处理结果
                if overwrite:
                    resized_img.save(file_path)
                else:
                    base, ext = os.path.splitext(filename)
                    new_path = os.path.join(mask_folder, f"{base}_processed{ext}")
                    resized_img.save(new_path)

        except Exception as e:
            print(f"处理 {filename} 时出错：{str(e)}")
            continue

    print(f"掩码预处理完成：{mask_folder}")


def batch_convert_datasets(root_data_dir, output_root):
    """批量处理所有数据集"""
    os.makedirs(output_root, exist_ok=True)

    # 遍历每个子数据集目录
    for dataset_dir in os.listdir(root_data_dir):
        dataset_path = os.path.join(root_data_dir, dataset_dir)

        if os.path.isdir(dataset_path):
            print(f"\n发现数据集目录：{dataset_dir}")
            valid_datasets = find_all_datasets(dataset_path)

            if not valid_datasets:
                print(f"目录 {dataset_dir} 中没有有效数据集")
                continue

            # 为每个数据集创建输出目录
            dataset_output_dir = os.path.join(output_root, dataset_dir)
            os.makedirs(dataset_output_dir, exist_ok=True)

            # 处理每个子数据集
            for dataset_name in valid_datasets:
                try:
                    process_single_dataset(dataset_path, dataset_output_dir, dataset_name)
                except Exception as e:
                    print(f"处理 {dataset_name} 时出错：{str(e)}")
                    continue


if __name__ == "__main__":
    # 配置路径（根据实际情况修改）
    ROOT_DATA_DIR = r'E:\CAGS-Net\data'
    OUTPUT_ROOT = r'E:\CAGS-Net\data'

    # 执行转换
    batch_convert_datasets(ROOT_DATA_DIR, OUTPUT_ROOT)
    print("\n所有数据集处理完成！")