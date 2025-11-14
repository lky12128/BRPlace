import os
from os.path import join, exists, splitext
import numpy as np
import cv2
import torch
import torch.utils.data as data
import h5py
import faiss

try:
    from bevdata_dataset_for_radar import bevdata_train_seq, bevdata_val_seq, bevdata_test_seq
    from bevdata_dataset_for_radar import extract_timestamp, evaluateResults 
except ImportError:
    print("警告: 无法导入 'bevdata_dataset.py'。将使用本地定义的字典。")
    # (如果导入失败，使用备用定义)
    bevdata_train_seq =  {"query": [], "db": []}
    bevdata_val_seq = {"query": [], "db": []}
    bevdata_test_seq = {"query": [], "db": []}
    def extract_timestamp(filename): # (简易版)
        base_name = splitext(os.path.basename(filename))[0]
        return float(base_name)
    def evaluateResults(*args, **kwargs):
        print("警告: 正在使用虚拟 evaluateResults 函数！")
        return 0.0, 0.0, 0.0, 0.0

# -----------------------------------------------------------------------------
# RangeProjection 类
# -----------------------------------------------------------------------------
class RangeProjection(object):
    def __init__(self, fov_up, fov_down, proj_w, proj_h, fov_left, fov_right):
        assert fov_up >= 0 and fov_down <= 0, 'require fov_up >= 0 and fov_down <= 0'
        assert fov_right >= 0 and fov_left <= 0, 'require fov_right >= 0 and fov_left <= 0'
        
        self.fov_up = fov_up / 180.0 * np.pi
        self.fov_down = fov_down / 180.0 * np.pi
        self.fov_v = abs(self.fov_up) + abs(self.fov_down)

        self.fov_left = fov_left / 180.0 * np.pi
        self.fov_right = fov_right / 180.0 * np.pi
        self.fov_h = abs(self.fov_left) + abs(self.fov_right)

        self.proj_w = proj_w
        self.proj_h = proj_h

    def doProjection(self, pointcloud: np.ndarray):
        points_xyzd = pointcloud[:, :4] 
        depth = np.linalg.norm(points_xyzd[:, :3], 2, axis=1) + 1e-5
        x = points_xyzd[:, 0]
        y = points_xyzd[:, 1]
        z = points_xyzd[:, 2]
        yaw = -np.arctan2(y, x)
        pitch = np.arcsin(z / depth)
        proj_x = (yaw - self.fov_left) / self.fov_h
        proj_y = 1.0 - (pitch + abs(self.fov_down)) / self.fov_v
        proj_x *= self.proj_w
        proj_y *= self.proj_h
        proj_x = np.maximum(np.minimum(self.proj_w - 1, np.floor(proj_x)), 0).astype(np.int32)
        proj_y = np.maximum(np.minimum(self.proj_h - 1, np.floor(proj_y)), 0).astype(np.int32)
        order = np.argsort(depth)[::-1]
        depth = depth[order]
        points_xyzd = points_xyzd[order] 
        proj_y = proj_y[order]
        proj_x = proj_x[order]
        proj_range = np.full((self.proj_h, self.proj_w), 0, dtype=np.float32) 
        proj_range[proj_y, proj_x] = depth
        proj_pointcloud = np.full(
            (self.proj_h, self.proj_w, points_xyzd.shape[1]), 0, dtype=np.float32) 
        proj_pointcloud[proj_y, proj_x] = points_xyzd
        proj_idx = np.full((self.proj_h, self.proj_w), -1, dtype=np.int32)
        proj_idx[proj_y, proj_x] = np.arange(len(points_xyzd))
        proj_mask = (proj_idx > -1).astype(np.float32) 
        return proj_pointcloud, proj_range, proj_mask


class InferDataset(data.Dataset):
    # ==================== (修改) ====================
    def __init__(self, bev_seq, 
                 range_seq,
                 bev_data_path, 
                 range_data_path, 
                 range_data_type='eagleg7_npy', 
                 sample_inteval=1):
    # ================================================
        super().__init__()
        self.sample_inteval = sample_inteval
        self.bev_seq = bev_seq
        self.range_seq = range_seq

        # 1. (修改) 设置 BEV 路径 (使用 bev_data_path)
        self.bev_img_dir = join(bev_data_path, 'val', self.bev_seq, 'images')
        if not exists(self.bev_img_dir):
            raise FileNotFoundError(f"BEV 图像目录不存在: {self.bev_img_dir}")
            
        # 2. (修改) 设置 Range NPY 路径 (使用 range_data_path)
        # 假设: range_data_path 指向 Snail 根目录 (e.g., ./datasets/snail_radar)
        self.range_npy_dir = join(range_data_path, self.range_seq, range_data_type)
        if not exists(self.range_npy_dir):
             # 备用路径: 尝试在 bev_data_path 内部 (如果用户把所有东西放一起了)
            alt_range_dir = join(bev_data_path, 'val', self.bev_seq, range_data_type)
            if exists(alt_range_dir):
                self.range_npy_dir = alt_range_dir
            else:
                raise FileNotFoundError(f"Range NPY 目录不存在: {self.range_npy_dir} (也未在 {alt_range_dir} 找到)")
        print(f"[{self.bev_seq}] Range NPY 目录: {self.range_npy_dir}")

        # 3. 读取BEV图像文件 (逻辑不变)
        img_paths_all = [join(self.bev_img_dir, f) for f in os.listdir(self.bev_img_dir) 
                                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not img_paths_all:
            raise FileNotFoundError(f"BEV 目录 {self.bev_img_dir} 中无图像文件")
        
        img_timestamps = [extract_timestamp(p) for p in img_paths_all]
        sorted_idx = np.argsort(img_timestamps)
        img_paths_sorted = [img_paths_all[i] for i in sorted_idx]
        self.img_filenames_sorted = [os.path.basename(p) for p in img_paths_sorted]

        # 4. 抽样文件名 (逻辑不变)
        self.img_filenames = self.img_filenames_sorted[::self.sample_inteval]
        if not self.img_filenames:
            raise ValueError(f"抽样间隔 {self.sample_inteval} 过大，无图像剩余")

        # 5. (修改) 定义Pose目录 (使用 bev_data_path)
        self.pose_dir = join(bev_data_path, 'val', self.bev_seq, 'poses')
        if not exists(self.pose_dir):
            raise FileNotFoundError(f"Pose目录不存在: {self.pose_dir}")

        # 6. 匹配图像与Pose (逻辑不变)
        self.poses = self._match_pose_to_image() 

        if len(self.poses) != len(self.img_filenames):
            raise ValueError(f"匹配后Pose数量（{len(self.poses)}）与图像数量（{len(self.img_filenames)}）不匹配")

        # 7. (新增) 初始化投影器和归一化参数
        self._init_projector_and_norm()

    def _init_projector_and_norm(self):
        # (同 V3)
        self.projector = RangeProjection(
            fov_up=22.5, fov_down=-22.5,
            fov_left=-56.5, fov_right=56.5,
            proj_h=32, proj_w=512
        )
        self.proj_img_mean = torch.tensor([12.12, 0.92, -3.32, -1.04, 0.21], dtype=torch.float)
        self.proj_img_stds = torch.tensor([12.32, 12.43, 7.18, 0.86, 0.16], dtype=torch.float)
    
    def _match_pose_to_image(self):
        # (同 V3, 逻辑不变)
        pose_files = [f for f in os.listdir(self.pose_dir) if f.lower().endswith('_txt')]
        if not pose_files: raise FileNotFoundError(f"Pose目录 {self.pose_dir} 中无 _txt 后缀的文件")
        pose_info = []
        for f in pose_files:
            f_path = join(self.pose_dir, f)
            try:
                ts = extract_timestamp(f)
                pose_info.append( (ts, f, f_path) )
            except ValueError: pass
        if not pose_info: raise ValueError(f"Pose目录 {self.pose_dir} 中无有效时间戳文件")
        pose_info.sort(key=lambda x: x[0])
        pose_timestamps = [x[0] for x in pose_info]
        pose_paths = [x[2] for x in pose_info]
        img_info = []
        for img_filename in self.img_filenames: 
            try:
                img_ts = extract_timestamp(img_filename)
                img_info.append( (img_ts, img_filename) )
            except ValueError as e:
                raise ValueError(f"❌ 图像文件 {img_filename} 时间戳提取失败: {str(e)}")
        img_timestamps = [x[0] for x in img_info]
        poses = []
        print(f"\n=== [Fusion] 序列 {self.bev_seq}: 图像-Pose匹配结果 ===")
        for i, (img_ts, img_filename) in enumerate(img_info):
            closest_idx = np.argmin(np.abs(np.array(pose_timestamps) - img_ts))
            with open(pose_paths[closest_idx], 'r') as f:
                pose = np.array(f.readline().strip().split(), dtype=np.float32)
                poses.append(pose)
        return np.array(poses)

    def _load_bev_and_range(self, filename_png):
        
        # 1. 加载 BEV
        bev_path = join(self.bev_img_dir, filename_png)
        bev_img = cv2.imread(bev_path, 0)
        if bev_img is None: raise FileNotFoundError(f"无法读取BEV图像: {bev_path}")
        bev_img = bev_img.astype(np.float32)
        bev_img = bev_img[np.newaxis, :, :].repeat(3, 0)
        
        # 2. 加载 Range (.npy 点云)
        filename_npy = splitext(filename_png)[0] + '.npy'
        range_path = join(self.range_npy_dir, filename_npy)
        try:
            pointcloud = np.load(range_path)
        except FileNotFoundError:
             raise FileNotFoundError(f"无法找到Range NPY点云: {range_path}")
        
        # 3. 实时投影
        proj_pointcloud, proj_range, proj_mask = self.projector.doProjection(pointcloud)
        proj_range_tensor = torch.from_numpy(proj_range)
        proj_xyz_tensor = torch.from_numpy(proj_pointcloud[..., :3])
        proj_doppler_tensor = torch.from_numpy(proj_pointcloud[..., 3])
        
        range_tensor = torch.cat(
            [proj_range_tensor.unsqueeze(0), 
             proj_xyz_tensor.permute(2, 0, 1), 
             proj_doppler_tensor.unsqueeze(0)], 
            0)
        
        # 4. 归一化
        proj_mask_tensor = torch.from_numpy(proj_mask)
        range_tensor = (range_tensor - self.proj_img_mean[:, None, None]) / self.proj_img_stds[:, None, None]
        range_tensor = range_tensor * proj_mask_tensor.unsqueeze(0).float()

        bev_tensor = torch.from_numpy(bev_img).float() / 255.0 # 归一化到 [0, 1]
        return bev_tensor, range_tensor.float()

    def __getitem__(self, index):
        # (同 V3, 逻辑不变)
        img_filename_png = self.img_filenames[index]
        bev_img, range_img = self._load_bev_and_range(img_filename_png)
        return (bev_img, range_img), index

    def __len__(self):
        return len(self.img_filenames)


class TrainingDataset(data.Dataset):
    # ==================== (修改) ====================
    def __init__(self, bev_seq, 
                    range_seq,
                 bev_data_path, 
                 range_data_path, 
                 range_data_type='eagleg7_npy', 
                 max_frames=3000,
                 cache_path=None):
    # ================================================
        super().__init__()
        self.bev_seq = bev_seq
        self.range_seq = range_seq
        self.max_frames = max_frames
        
        # 1. (修改) 设置 BEV 路径
        self.bev_img_dir = join(bev_data_path, 'train', self.bev_seq, 'images')
        if not exists(self.bev_img_dir):
            raise FileNotFoundError(f"BEV 图像目录不存在: {self.bev_img_dir}")
            
        # 2. (修改) 设置 Range NPY 路径
        self.range_npy_dir = join(range_data_path, self.range_seq, range_data_type)
        if not exists(self.range_npy_dir):
            alt_range_dir = join(bev_data_path, 'train', self.bev_seq, range_data_type)
            if exists(alt_range_dir):
                self.range_npy_dir = alt_range_dir
            else:
                raise FileNotFoundError(f"Range NPY 目录不存在: {self.range_npy_dir}")
        print(f"[{self.bev_seq}] Range NPY 目录: {self.range_npy_dir}")

        # 3. 读取BEV图像文件 (逻辑不变)
        img_paths_all = [join(self.bev_img_dir, f) for f in os.listdir(self.bev_img_dir) 
                                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not img_paths_all:
            raise FileNotFoundError(f"BEV 目录 {self.bev_img_dir} 中无图像文件")
        
        img_timestamps = [extract_timestamp(p) for p in img_paths_all]
        sorted_idx = np.argsort(img_timestamps)
        self.img_filenames_sorted = [os.path.basename(img_paths_all[i]) for i in sorted_idx]

        # 4. (修改) 读取pose文件 (使用 bev_data_path)
        self.pose_dir = join(bev_data_path, 'train', self.bev_seq, 'poses')
        # (复制 bevdata_dataset.py 中的 pose 读取)
        # --- (开始) 复制 ---
        if not exists(self.pose_dir):
            raise FileNotFoundError(f"Pose目录不存在: {self.pose_dir}")
        self.pose_paths_all = [join(self.pose_dir, f) for f in os.listdir(self.pose_dir) 
                                   if f.lower().endswith('_txt')]
        if not self.pose_paths_all:
            raise FileNotFoundError(f"Pose目录 {self.pose_dir} 中无 _txt 后缀的文件")
        self.pose_timestamps = []
        self.poses_all = []
        for p_path in self.pose_paths_all:
            try:
                ts = extract_timestamp(p_path)
                pose = np.loadtxt(p_path, dtype=np.float32)
                if pose.shape != (12,):
                    raise ValueError(f"Pose文件 {p_path} 格式错误")
                self.pose_timestamps.append(ts)
                self.poses_all.append(pose)
            except Exception as e:
                raise RuntimeError(f"读取Pose文件 {p_path} 失败: {str(e)}")
        self.pose_timestamps = np.array(self.pose_timestamps)
        self.poses_all = np.array(self.poses_all)
        # --- (结束) 复制 ---

        # 5. 图像与pose匹配并截取帧 (逻辑不变)
        self.img_filenames, self.poses = self._match_and_truncate_frames()
        if len(self.poses) == 0:
            raise ValueError(f"匹配后无有效帧，或超过最大帧数 {self.max_frames}")

        # 6. 计算正负样本 (逻辑不变)
        self.pos_thres = 10
        self.neg_thres = 100
        self.num_neg = 10
        self.positives, self.negatives = self._compute_pos_neg_samples()

        self.mining = False
        if cache_path is not None:
            self.cache = join(cache_path, 'desc_cen_hardmining_fusion.hdf5')
        else:
            self.cache = None  # 避免默认值导致的路径错误
        
        # 7. (新增) 初始化投影器和归一化参数
        self._init_projector_and_norm()

    def _init_projector_and_norm(self):
        # (同 V3)
        self.projector = RangeProjection(
            fov_up=22.5, fov_down=-22.5,
            fov_left=-56.5, fov_right=56.5,
            proj_h=32, proj_w=512
        )
        self.proj_img_mean = torch.tensor([12.12, 0.92, -3.32, -1.04, 0.21], dtype=torch.float)
        self.proj_img_stds = torch.tensor([12.32, 12.43, 7.18, 0.86, 0.16], dtype=torch.float)

    def _match_and_truncate_frames(self):
        # (同 V3)
        img_timestamps = np.array([extract_timestamp(f) for f in self.img_filenames_sorted])
        pose_timestamps = self.pose_timestamps
        poses_matched = []
        for img_ts in img_timestamps:
            ts_diff = np.abs(pose_timestamps - img_ts)
            best_pose_idx = np.argmin(ts_diff)
            poses_matched.append(self.poses_all[best_pose_idx])
        poses_matched = np.array(poses_matched)

        truncate_num = min(self.max_frames, len(self.img_filenames_sorted))
        img_filenames_truncated = self.img_filenames_sorted[:truncate_num]
        poses_truncated = poses_matched[:truncate_num]

        return img_filenames_truncated, poses_truncated

    def _compute_pos_neg_samples(self):
        # (同 V3)
        positives = []
        negatives = []
        num_frames = len(self.poses)
        all_poses = self.poses
        for qi in range(num_frames):
            q_pose = all_poses[qi]
            trans_diff = (all_poses - q_pose)[:, [3, 7, 11]]
            trans_dist = np.sqrt(np.sum(trans_diff ** 2, axis=1))
            pos_idx = np.where((trans_dist < self.pos_thres) & (np.arange(num_frames) != qi))[0]
            positives.append(pos_idx)
            neg_idx = np.where(trans_dist > self.neg_thres)[0]
            negatives.append(neg_idx)
        for qi in range(num_frames):
            if len(positives[qi]) == 0:
                raise ValueError(f"帧 {qi} 无正样本")
            if len(negatives[qi]) < self.num_neg:
                raise ValueError(f"帧 {qi} 负样本不足")
        return positives, negatives

    def refreshCache(self):
        # (同 V3)
        if self.cache is None:
            raise ValueError("开启硬样本挖掘前，请先设置 self.cache 为HDF5特征文件路径！")
        with h5py.File(self.cache, 'r') as h5:
            self.h5feat = np.array(h5.get("features"))
            if self.h5feat.shape[0] != len(self.poses):
                raise ValueError(f"HDF5特征数量与数据集帧数不匹配！")

    def _load_images_with_aug(self, filename_png):
        """(同 V3) 辅助函数: 加载 BEV (.png) 和 Range (.npy) 并应用增强"""
        
        # 1. 加载 BEV
        bev_path = join(self.bev_img_dir, filename_png)
        bev_img = cv2.imread(bev_path, 0)
        if bev_img is None:
            raise FileNotFoundError(f"无法读取BEV图像: {bev_path}")
        
        # rot_mat = cv2.getRotationMatrix2D((bev_img.shape[1]//2, bev_img.shape[0]//2), np.random.randint(0, 360), 1.0)
        # bev_img = cv2.warpAffine(bev_img, rot_mat, bev_img.shape[:2])
        bev_img = bev_img.astype(np.float32)[np.newaxis, :, :].repeat(3, 0)
        
        # 2. 加载 Range (.npy 点云)
        filename_npy = splitext(filename_png)[0] + '.npy'
        range_path = join(self.range_npy_dir, filename_npy)
        try:
            pointcloud = np.load(range_path)
        except FileNotFoundError:
             raise FileNotFoundError(f"无法找到Range NPY点云: {range_path}")
        
        # 3. 实时投影
        proj_pointcloud, proj_range, proj_mask = self.projector.doProjection(pointcloud)
        
        proj_range_tensor = torch.from_numpy(proj_range)
        proj_xyz_tensor = torch.from_numpy(proj_pointcloud[..., :3]) 
        proj_doppler_tensor = torch.from_numpy(proj_pointcloud[..., 3])
        
        range_tensor = torch.cat(
            [proj_range_tensor.unsqueeze(0), 
             proj_xyz_tensor.permute(2, 0, 1), 
             proj_doppler_tensor.unsqueeze(0)], 
            0)
        
        # 4. 归一化
        proj_mask_tensor = torch.from_numpy(proj_mask)
        range_tensor = (range_tensor - self.proj_img_mean[:, None, None]) / self.proj_img_stds[:, None, None]
        range_tensor = range_tensor * proj_mask_tensor.unsqueeze(0).float()
        
        bev_tensor = torch.from_numpy(bev_img).float() / 255.0 # 归一化到 [0, 1]
        return bev_tensor, range_tensor.float()


    def __getitem__(self, index):
        # 1. 计算正负样本索引（mining=True时选硬样本，否则随机选）
        if self.mining:
            if not hasattr(self, 'h5feat'):
                raise RuntimeError("开启mining前请先调用 refreshCache() 加载特征！")
            # 硬正样本：选择距离查询样本最远的正样本
            pos_idx_list = self.positives[index]
            q_feat = self.h5feat[index]
            pos_feats = self.h5feat[pos_idx_list]
            pos_dist = np.sqrt(np.sum((q_feat - pos_feats) **2, axis=1))
            pos_idx = pos_idx_list[np.argmax(pos_dist)]
            
            # 硬负样本：选择距离查询样本最近的负样本
            neg_idx_list = self.negatives[index]
            neg_feats = self.h5feat[neg_idx_list]
            neg_dist = np.sqrt(np.sum((q_feat - neg_feats)** 2, axis=1))
            neg_idx = neg_idx_list[np.argsort(neg_dist)[:self.num_neg]]
        else:
            # 非mining模式（如构建缓存）：随机选择正负样本（仅用于缓存，不影响训练）
            pos_idx = np.random.choice(self.positives[index], 1)[0]
            neg_idx = np.random.choice(self.negatives[index], self.num_neg, replace=False)

        # 2. 加载查询、正样本、负样本的数据
        query_filename = self.img_filenames[index]
        query_bev, query_range = self._load_images_with_aug(query_filename)

        pos_filename = self.img_filenames[int(pos_idx)]
        positive_bev, positive_range = self._load_images_with_aug(pos_filename)

        negatives_bev = []
        negatives_range = []
        for ni in neg_idx:
            neg_filename = self.img_filenames[int(ni)]
            negative_bev, negative_range = self._load_images_with_aug(neg_filename)
            negatives_bev.append(negative_bev)
            negatives_range.append(negative_range)
        negatives_bev = torch.stack(negatives_bev, 0)
        negatives_range = torch.stack(negatives_range, 0)

        # 3. 根据模式返回不同数量的元素
        if self.mining:
            # 训练阶段：返回4个元素（查询, 正样本, 负样本, 索引）
            return (query_bev, query_range), (positive_bev, positive_range), (negatives_bev, negatives_range), index
        else:
            # 缓存构建阶段：仅返回查询数据和索引（不需要正负样本）
            return (query_bev, query_range), index

    def __len__(self):
        return len(self.poses)


def collate_fn(batch):
    """处理训练阶段的样本，每个样本为4元素元组"""
    batch = list(filter(lambda x: x is not None, batch))  # 过滤无效样本
    if len(batch) == 0:
        return None, None, None, None
    
    # 解压4个元素：查询数据、正样本数据、负样本数据、索引
    query_list, positive_list, negatives_list, indices = zip(*batch)
    
    # 处理查询数据（ bev 和 range 分别堆叠）
    query_bev, query_range = zip(*query_list)
    query_bev = torch.stack(query_bev, 0)
    query_range = torch.stack(query_range, 0)
    
    # 处理正样本数据
    positive_bev, positive_range = zip(*positive_list)
    positive_bev = torch.stack(positive_bev, 0)
    positive_range = torch.stack(positive_range, 0)
    
    # 处理负样本数据（每个样本的负样本已堆叠，这里只需在batch维度拼接）
    negatives_bev, negatives_range = zip(*negatives_list)
    negatives_bev = torch.cat(negatives_bev, dim=0)  # shape: [batch_size * num_neg, ...]
    negatives_range = torch.cat(negatives_range, dim=0)
    
    return (query_bev, query_range), (positive_bev, positive_range), (negatives_bev, negatives_range), indices