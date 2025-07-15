import os, json, torch
import torchvision.transforms.functional as F

from torch.utils.data import Dataset
from PIL import Image
from collections import defaultdict
from typing import Literal, Any, Union, Callable

from ..utils.api import GroundTruth


class TAOAmodalDataset(Dataset):
    """
    TAO-Amodal 数据集加载器。

    该加载器为基于视频切片的迭代训练方法设计。
    它会加载一个由 `num_frames` 和 `num_iter` 定义的视频片段。

    参数:
    - root_dir (str): 数据集根目录，应包含 'frames' 和 'annotations' 子目录。
    - ann_file (str): 标注文件的路径 (例如: 'amodal_annotations/amodal_train_coco.json')。
    - num_frames (int): 模型一次输入的基础帧数 (T)。
    - frame_interval (int): 采样视频帧时的时间间隔。默认为 1。
    - transform (callable, optional): 应用于图像张量的数据增强/转换。默认为 None。
    """
    def __init__(self, root_dir: str, ann_file: str, num_frames: int, frame_interval: int = 1, mode: Literal["train", "val"] = "train", transform=None):
        super().__init__()
        
        self.root_dir = root_dir
        self.frames_dir = os.path.join(self.root_dir, 'frames', mode)
        self.num_frames = num_frames
        self.frame_interval = frame_interval
        self.transform = transform

        print("正在加载标注文件...")
        with open(os.path.join(root_dir, ann_file), 'r') as f:
            self.coco_data = json.load(f)
        print("标注文件加载完成。")

        self._process_annotations()
        
        # 初始化时使用默认的迭代次数
        self.set_iter_num(num_iter=1)

    def _process_annotations(self):
        """
        预处理标注信息，建立视频、图像和标注之间的映射关系。
        """
        # 按视频ID对图像进行分组
        self.imgs_by_vid = defaultdict(list)
        for img in self.coco_data['images']:
            self.imgs_by_vid[img['video_id']].append(img)

        # 对每个视频的图像按帧索引排序
        for vid in self.imgs_by_vid:
            self.imgs_by_vid[vid].sort(key=lambda x: x.get('frame_index', 0)) # 确保帧是按顺序的

        # 按图像ID对标注进行分组，方便快速查找
        self.anns_by_img = defaultdict(list)
        if 'annotations' in self.coco_data:
            for ann in self.coco_data['annotations']:
                self.anns_by_img[ann['image_id']].append(ann)

    def set_iter_num(self, num_iter: int):
        """
        设置迭代训练的轮数，并重新计算有效样本。
        这个方法应该在每个epoch开始前或者需要改变迭代次数时调用。

        参数:
        - num_iter (int): 训练迭代的轮数。
        """
        print(f"设置 num_iter 为 {num_iter}，正在重新生成样本列表...")
        self.num_iter = num_iter
        self.samples = []
        
        # 一个完整的训练样本需要加载的总帧数
        clip_len = self.num_frames + self.num_iter - 1
        if clip_len <= 0:
            print("警告: num_frames + num_iter - 1 小于等于 0，无法生成样本。")
            return

        # 遍历所有视频，生成有效的起始帧索引
        for vid, imgs in self.imgs_by_vid.items():
            video_len = len(imgs)
            # 计算此视频能产生多少个有效的切片
            # 一个切片需要的总跨度是 (clip_len - 1) * frame_interval
            # 所以最后一帧的索引是 start_idx + (clip_len - 1) * frame_interval
            # 这个索引必须小于 video_len
            num_possible_starts = video_len - (clip_len - 1) * self.frame_interval
            
            if num_possible_starts > 0:
                for i in range(num_possible_starts):
                    # 每个样本由 (视频ID, 在该视频帧列表中的起始索引) 组成
                    self.samples.append((vid, i))
        
        print(f"生成了 {len(self.samples)} 个有效样本。")


    def __len__(self) -> int:
        """
        返回数据集中有效样本（视频切片）的总数。
        """
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple:
        """
        获取一个训练样本（视频切片及其对应的标注）。

        参数:
        - idx (int): 样本索引。

        返回:
        - tuple: (images, ground_truths)
          - images (Tensor): [T, 3, H, W] 形状的图像张量。
          - ground_truths (list[dict]): 长度为 T 的列表，每个元素是该帧的标注字典。
        """
        # 1. 获取样本信息
        video_id, start_frame_idx = self.samples[idx]
        
        # 2. 计算需要加载的帧的索引
        clip_len = self.num_frames + self.num_iter - 1
        frame_indices_in_video = [start_frame_idx + i * self.frame_interval for i in range(clip_len)]
        
        images = []
        ground_truths = []
        
        video_frames = self.imgs_by_vid[video_id]

        # 3. 循环加载每一帧及其标注
        for frame_idx in frame_indices_in_video:
            img_info = video_frames[frame_idx]
            img_id = img_info['id']
            
            # 加载图像
            img_path = os.path.join(self.frames_dir, img_info['file_name'])
            image = Image.open(img_path).convert('RGB')
            W, H = image.size
            
            # 将PIL Image转换为Tensor
            images.append(F.to_tensor(image))
            
            # 加载并处理标注
            gt = {
                'labels': [],
                'boxes': [],
                'ids': [],
                'areas': []
            }
            
            if img_id in self.anns_by_img:
                for ann in self.anns_by_img[img_id]:
                    gt['labels'].append(ann['category_id'])
                    gt['ids'].append(ann.get('track_id', -1)) # 使用 .get 以防万一没有 track_id
                    gt['areas'].append(ann.get('area', 0.0))
                    
                    # 转换box格式: 从 [x, y, w, h] 到归一化的 [cx, cy, w, h]
                    x, y, w, h = ann['bbox']
                    cx = (x + w / 2) / W
                    cy = (y + h / 2) / H
                    norm_w = w / W
                    norm_h = h / H
                    gt['boxes'].append([cx, cy, norm_w, norm_h])

            # 将列表转换为Tensor
            gt['labels'] = torch.tensor(gt['labels'], dtype=torch.int64)
            gt['boxes'] = torch.tensor(gt['boxes'], dtype=torch.float32)
            gt['ids'] = torch.tensor(gt['ids'], dtype=torch.int64)
            gt['areas'] = torch.tensor(gt['areas'], dtype=torch.float32)
            gt['num_iter']: int = self.num_iter
            
            ground_truths.append(gt)

        # 4. 将图像列表堆叠成一个张量
        images_tensor = torch.stack(images) # Shape: [clip_len, 3, H, W]

        # 5. (可选) 应用数据增强
        if self.transform:
            # 注意：复杂的数据增强（如几何变换）需要同时作用于图像和box，
            # 此处仅为示例，假设transform只作用于图像。
            images_tensor = self.transform(images_tensor)
            
        return GroundTruth(images_tensor, ground_truths)


def collate_fn_tao(batch: list[GroundTruth]):
    return batch


# FIXME not implemented
def build_tao_transform() -> Callable[[Union[Image.Image, torch.Tensor], Any], tuple[Union[Image.Image, torch.Tensor], Any]]:
    return lambda img, label: (img, label)


if __name__ == '__main__':
    # --- 使用示例 ---
    # 假设你的数据集解压在 './TAO-Amodal' 目录下
    # 目录结构如下:
    # ./TAO-Amodal/
    #   ├── frames/
    #   │   ├── train/
    #   │   │   └── ArgoVerse/
    #   │   │       └── ... (video folders)
    #   │   └── val/
    #   └── amodal_annotations/
    #       └── amodal_train_coco.json
    
    # 1. 初始化参数
    DATASET_ROOT = './TAO-Amodal' # <-- 修改为你的数据集根目录
    ANN_FILE = 'amodal_annotations/amodal_train_coco.json' # <-- 训练集标注文件
    NUM_FRAMES = 5   # T
    FRAME_INTERVAL = 2 # 采样间隔

    # 2. 创建数据集实例
    # 检查路径是否存在，如果不存在则跳过示例执行
    if not os.path.exists(DATASET_ROOT) or not os.path.exists(os.path.join(DATASET_ROOT, ANN_FILE)):
        print("="*50)
        print("示例代码未执行：请将 'DATASET_ROOT' 修改为你的TAO-Amodal数据集根目录。")
        print(f"当前查找路径: {os.path.abspath(DATASET_ROOT)}")
        print("="*50)
    else:
        print("创建数据集实例...")
        dataset = TAOAmodalDataset(
            root_dir=DATASET_ROOT,
            ann_file=ANN_FILE,
            num_frames=NUM_FRAMES,
            frame_interval=FRAME_INTERVAL
        )

        # 默认情况下, num_iter=1
        print(f"\n默认 num_iter=1, 数据集大小: {len(dataset)}")
        
        # 模拟从数据集中取一个样本
        if len(dataset) > 0:
            images, gts = dataset[0]
            print("\n--- 获取一个样本 (num_iter=1) ---")
            print(f"图像张量形状: {images.shape}")
            print(f"标注列表长度: {len(gts)}")
            # 期望的图像张量形状: [5, 3, H, W] (因为 num_frames=5, num_iter=1)
            # 期望的标注列表长度: 5
            assert images.shape[0] == NUM_FRAMES + 1 - 1
            assert len(gts) == NUM_FRAMES + 1 - 1
            print("第一个样本的标注信息 (第一帧):")
            print(gts[0])
            print("--- 样本获取成功 ---\n")

        # 3. 模拟在训练过程中改变 num_iter
        print("="*20)
        print("模拟训练：改变 num_iter")
        print("="*20)
        new_num_iter = 4
        dataset.set_iter_num(new_num_iter)
        
        print(f"\n设置 num_iter={new_num_iter} 后, 数据集大小: {len(dataset)}")
        
        # 再次从数据集中取一个样本
        if len(dataset) > 0:
            images, gts = dataset[0]
            print(f"\n--- 获取一个样本 (num_iter={new_num_iter}) ---")
            print(f"图像张量形状: {images.shape}")
            print(f"标注列表长度: {len(gts)}")
            # 期望的图像张量形状: [8, 3, H, W] (因为 num_frames=5, num_iter=4 -> 5+4-1=8)
            # 期望的标注列表长度: 8
            assert images.shape[0] == NUM_FRAMES + new_num_iter - 1
            assert len(gts) == NUM_FRAMES + new_num_iter - 1
            print("第一个样本的标注信息 (第一帧):")
            print(gts[0])
            print("--- 样本获取成功 ---")

        # 4. 配合 DataLoader 使用
        from torch.utils.data import DataLoader
        
        # 确保数据集不为空
        if len(dataset) > 0:
            data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
            
            # 模拟一个批次
            batch_images, batch_gts = next(iter(data_loader))
            
            print("\n--- DataLoader 批次示例 ---")
            print(f"批次图像形状: {batch_images.shape}")
            # batch_gts 的结构会比较复杂，因为默认的 collate_fn 不会堆叠字典列表
            # 它会是一个长度为 batch_size 的列表，每个元素是 __getitem__ 返回的 gts 列表
            print(f"批次标注类型: {type(batch_gts)}")
            print(f"批次标注长度 (等于batch_size): {len(batch_gts)}")
            print(f"批次中第一个样本的标注列表长度: {len(batch_gts[0])}")
            print("--- DataLoader 示例结束 ---")