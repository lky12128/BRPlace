import os
from os.path import join, exists, splitext
import numpy as np
import cv2
from imgaug import augmenters as iaa
import torch
import torch.utils.data as data
import h5py
import faiss
from RANSAC import rigidRansac

# 数据集序列配置（保持不变）
bevdata_train_seq =   {"query": ['iaf_20231201_3','81r_20240116_eve_3'],
                        "db": ["iaf_20231201_2",'iaef_20240115_2']}
bevdata_val_seq =     {"query": ['iaf_20231201_3'],
                        "db": ["iaf_20231201_2"]}
bevdata_test_seq =     {"query": ['81r_20240116_2','81r_20240116_eve_3'],
                        "db": ["81r_20240123_2",'iaef_20240115_2']}
bevdata_train_pairs = [("ss_20231109_4", "ss_20231105_aft_5"),
                       ("if_20240116_5",  "if_20231208_4"),
                       ("sl_20231105_2",  "sl_20231105_aft_4")]
bevdata_train_split=1000

range_train_seq = {
    "db": ["20231201_2"],  
    "query": []
}
range_val_seq = {
    "db": ["20231201_2"],
    "query": ["20231201_3"]
}
range_test_seq = {
    "db": ["20231201_2"],
    "query": ["20231201_3"]
}


def extract_timestamp(filename):
    """从文件名提取时间戳（适配：
    1. 图像文件：纯数字（1705999984）、数字.小数（1705999984.964096291）；
    2. Pose文件：数字_小数_后缀（1705999985_487979_txt → 1705999985.487979）
    """
    base_name = splitext(os.path.basename(filename))[0]
    try:
        if '_' in base_name:
            parts = base_name.split('_', 1)
            if len(parts) != 2:
                raise ValueError(f"无法分割时间戳：{base_name}（需符合「数字_小数_后缀」格式）")
            decimal_part = parts[1].split('_')[0]
            if not decimal_part.isdigit():
                raise ValueError(f"小数部分非数字：{decimal_part}（文件名：{base_name}）")
            timestamp_str = f"{parts[0]}.{decimal_part}"
            return float(timestamp_str)
        else:
            return float(base_name)
    except (ValueError, IndexError) as e:
        raise ValueError(f"文件名 {filename} 无法提取时间戳：{str(e)}")


class InferDataset(data.Dataset):
    def __init__(self, seq, dataset_path='', data_type='lidar_bev_z', sample_inteval=1):
        super().__init__()
        self.sample_inteval = sample_inteval
        self.dataset_path = dataset_path  
        self.data_type = data_type  
        self.datapath_type = join(dataset_path, data_type, 'val')  # 验证集推理
        self.seq = seq

        # 1. 读取图像文件，提取时间戳并按时间排序
        self.img_dir = join(self.datapath_type, seq, 'images')
        if not exists(self.img_dir):
            raise FileNotFoundError(f"图像目录不存在: {self.img_dir}")
        self.img_paths_all = [join(self.img_dir, f) for f in os.listdir(self.img_dir) 
                              if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not self.img_paths_all:
            raise FileNotFoundError(f"图像目录 {self.img_dir} 中无图像文件")
        
        self.img_timestamps = [extract_timestamp(p) for p in self.img_paths_all]
        sorted_idx = np.argsort(self.img_timestamps)
        self.img_paths_sorted = [self.img_paths_all[i] for i in sorted_idx]
        self.img_timestamps_sorted = [self.img_timestamps[i] for i in sorted_idx]

        # 2. 抽样图像
        self.imgs_path = self.img_paths_sorted[::self.sample_inteval]
        self.imgs_timestamps_sampled = self.img_timestamps_sorted[::self.sample_inteval]
        if not self.imgs_path:
            raise ValueError(f"抽样间隔 {self.sample_inteval} 过大，无图像剩余")

        # 3. 定义Pose目录
        self.pose_dir = join(self.datapath_type, seq, 'poses')
        if not exists(self.pose_dir):
            raise FileNotFoundError(f"Pose目录不存在: {self.pose_dir}")

        # 4. 匹配图像与Pose
        self.poses = self._match_pose_to_image()

        # 验证匹配后数量一致
        if len(self.poses) != len(self.imgs_path):
            raise ValueError(f"匹配后Pose数量（{len(self.poses)}）与图像数量（{len(self.imgs_path)}）不匹配")

    def _match_pose_to_image(self):
        pose_files = [f for f in os.listdir(self.pose_dir) if f.lower().endswith('_txt')]
        if not pose_files:
            raise FileNotFoundError(f"Pose目录 {self.pose_dir} 中无 _txt 后缀的文件")

        pose_info = []
        for f in pose_files:
            f_path = join(self.pose_dir, f)
            try:
                ts = extract_timestamp(f)
                pose_info.append( (ts, f, f_path) )
            except ValueError as e:
                print(f"⚠️  Pose文件 {f} 时间戳提取失败: {str(e)}")
        
        if not pose_info:
            raise ValueError(f"Pose目录 {self.pose_dir} 中无有效时间戳文件")
        
        pose_info.sort(key=lambda x: x[0])
        pose_timestamps = [x[0] for x in pose_info]
        pose_paths = [x[2] for x in pose_info]

        img_info = []
        for img_path in self.imgs_path:
            img_filename = os.path.basename(img_path)
            try:
                img_ts = extract_timestamp(img_path)
                img_info.append( (img_ts, img_filename, img_path) )
            except ValueError as e:
                raise ValueError(f"❌ 图像文件 {img_filename} 时间戳提取失败: {str(e)}")
        
        img_timestamps = [x[0] for x in img_info]

        poses = []
        print("\n=== 图像-Pose匹配结果（前10组 + 最后10组）===")
        print(f"{'图像文件名':<20} {'图像时间戳':<15} {'匹配的Pose文件名':<25} {'Pose时间戳':<15} {'时间差(秒)':<10}")
        print("-" * 90)

        for i, (img_ts, img_filename, _) in enumerate(img_info):
            closest_idx = np.argmin(np.abs(np.array(pose_timestamps) - img_ts))
            closest_pose_ts = pose_timestamps[closest_idx]
            closest_pose_filename = os.path.basename(pose_paths[closest_idx])
            time_diff = abs(img_ts - closest_pose_ts)

            if i < 10 or i >= len(img_info) - 10:
                print(f"{img_filename:<20} {img_ts:<15.2f} {closest_pose_filename:<25} {closest_pose_ts:<15.2f} {time_diff:<10.2f}")
            
            with open(pose_paths[closest_idx], 'r') as f:
                pose = np.array(f.readline().strip().split(), dtype=np.float32)
                if len(pose) != 12:
                    raise ValueError(f"❌ Pose文件 {closest_pose_filename} 格式错误，需12个数值（当前{len(pose)}个）")
                poses.append(pose)

        all_time_diffs = [abs(img_ts - pose_timestamps[np.argmin(np.abs(np.array(pose_timestamps) - img_ts))]) 
                        for img_ts in img_timestamps]
        max_diff = max(all_time_diffs)
        avg_diff = np.mean(all_time_diffs)
        print("-" * 90)
        print(f"时间差统计：最大 {max_diff:.2f} 秒 | 平均 {avg_diff:.2f} 秒")
        if max_diff > 1.0:
            print(f"⚠️  警告：存在时间差超过1秒的匹配，大概率图像与Pose不匹配！")

        return np.array(poses)

    def __getitem__(self, index):
        img_path = self.imgs_path[index]
        img = cv2.imread(img_path, 0)
        if img is None:
            raise FileNotFoundError(f"无法读取图像（文件损坏或路径错误）: {img_path}")
        
        img = img.astype(np.float32)
        img = img[np.newaxis, :, :].repeat(3, 0)
        return img, index

    def __len__(self):
        return len(self.imgs_path)


class TrainingDataset(data.Dataset):
    def __init__(self, dataset_path='', data_type='', seq='', max_frames=3000):
        super().__init__()
        self.dataset_path = dataset_path
        self.seq = seq
        self.data_type = data_type
        self.datapath_type = join(dataset_path, data_type, 'train')
        self.max_frames = max_frames

        # 1. 读取图像文件
        self.img_dir = join(self.datapath_type, seq, 'images')
        if not exists(self.img_dir):
            raise FileNotFoundError(f"图像目录不存在: {self.img_dir}")
        self.img_paths_all = [join(self.img_dir, f) for f in os.listdir(self.img_dir) 
                              if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not self.img_paths_all:
            raise FileNotFoundError(f"图像目录 {self.img_dir} 中无图像文件")
        
        self.img_timestamps = [extract_timestamp(p) for p in self.img_paths_all]
        sorted_idx = np.argsort(self.img_timestamps)
        self.img_paths_sorted = [self.img_paths_all[i] for i in sorted_idx]
        self.img_timestamps_sorted = [self.img_timestamps[i] for i in sorted_idx]

        # 2. 读取pose文件
        self.pose_dir = join(self.datapath_type, seq, 'poses')
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
                    raise ValueError(f"Pose文件 {p_path} 格式错误（需12个数值），实际形状: {pose.shape}")
                self.pose_timestamps.append(ts)
                self.poses_all.append(pose)
            except Exception as e:
                raise RuntimeError(f"读取Pose文件 {p_path} 失败: {str(e)}")
        self.pose_timestamps = np.array(self.pose_timestamps)
        self.poses_all = np.array(self.poses_all)

        # 3. 图像与pose匹配并截取帧
        self.imgs_path, self.poses = self._match_and_truncate_frames()
        if len(self.poses) == 0:
            raise ValueError(f"匹配后无有效帧，或超过最大帧数 {self.max_frames}")

        # 4. 计算正负样本
        self.pos_thres = 10
        self.neg_thres = 50
        self.num_neg = 10
        self.positives, self.negatives = self._compute_pos_neg_samples()

        self.mining = False
        self.cache = None

    def _match_and_truncate_frames(self):
        img_timestamps = np.array(self.img_timestamps_sorted)
        pose_timestamps = self.pose_timestamps
        poses_matched = []
        for img_ts in img_timestamps:
            ts_diff = np.abs(pose_timestamps - img_ts)
            best_pose_idx = np.argmin(ts_diff)
            poses_matched.append(self.poses_all[best_pose_idx])
        poses_matched = np.array(poses_matched)

        truncate_num = min(self.max_frames, len(self.img_paths_sorted))
        imgs_path_truncated = self.img_paths_sorted[:truncate_num]
        poses_truncated = poses_matched[:truncate_num]

        return imgs_path_truncated, poses_truncated

    def _compute_pos_neg_samples(self):
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
                raise ValueError(f"帧 {qi} 无正样本（建议调大pos_thres，当前{self.pos_thres}米）")
            if len(negatives[qi]) < self.num_neg:
                raise ValueError(f"帧 {qi} 负样本不足（需{self.num_neg}个，实际{len(negatives[qi])}个，建议调小neg_thres，当前{self.neg_thres}米）")
        
        return positives, negatives

    def refreshCache(self):
        if self.cache is None:
            raise ValueError("开启硬样本挖掘前，请先设置 self.cache 为HDF5特征文件路径！")
        with h5py.File(self.cache, 'r') as h5:
            self.h5feat = np.array(h5.get("features"))
            if self.h5feat.shape[0] != len(self.poses):
                raise ValueError(f"HDF5特征数量（{self.h5feat.shape[0]}）与数据集帧数（{len(self.poses)}）不匹配！")

    def __getitem__(self, index):
        if self.mining:
            if not hasattr(self, 'h5feat'):
                raise RuntimeError("开启mining前请先调用 refreshCache() 加载特征！")
            pos_idx_list = self.positives[index]
            q_feat = self.h5feat[index]
            pos_feats = self.h5feat[pos_idx_list]
            pos_dist = np.sqrt(np.sum((q_feat - pos_feats) ** 2, axis=1))
            pos_idx = pos_idx_list[np.argmax(pos_dist)]
            
            neg_idx_list = self.negatives[index]
            neg_feats = self.h5feat[neg_idx_list]
            neg_dist = np.sqrt(np.sum((q_feat - neg_feats) ** 2, axis=1))
            neg_idx = neg_idx_list[np.argsort(neg_dist)[:self.num_neg]]
        else:
            pos_idx = np.random.choice(self.positives[index], 1)[0]
            neg_idx = np.random.choice(self.negatives[index], self.num_neg, replace=False)

        # 读取查询图像
        query_path = self.imgs_path[index]
        query = cv2.imread(query_path, 0)
        if query is None:
            raise FileNotFoundError(f"无法读取查询图像: {query_path}")
        rot_mat = cv2.getRotationMatrix2D((query.shape[1]//2, query.shape[0]//2), np.random.randint(0, 360), 1.0)
        query = cv2.warpAffine(query, rot_mat, query.shape[:2])
        query = query.astype(np.float32)[np.newaxis, :, :].repeat(3, 0)

        # 读取正样本图像
        pos_path = self.imgs_path[int(pos_idx)]
        positive = cv2.imread(pos_path, 0)
        if positive is None:
            raise FileNotFoundError(f"无法读取正样本图像: {pos_path}")
        rot_mat = cv2.getRotationMatrix2D((positive.shape[1]//2, positive.shape[0]//2), np.random.randint(0, 360), 1.0)
        positive = cv2.warpAffine(positive, rot_mat, positive.shape[:2])
        positive = positive.astype(np.float32)[np.newaxis, :, :].repeat(3, 0)

        # 读取负样本图像
        negatives = []
        for ni in neg_idx:
            neg_path = self.imgs_path[int(ni)]
            negative = cv2.imread(neg_path, 0)
            if negative is None:
                raise FileNotFoundError(f"无法读取负样本图像: {neg_path}")
            rot_mat = cv2.getRotationMatrix2D((negative.shape[1]//2, negative.shape[0]//2), np.random.randint(0, 360), 1.0)
            negative = cv2.warpAffine(negative, rot_mat, negative.shape[:2])
            negative = negative.astype(np.float32)[np.newaxis, :, :].repeat(3, 0)
            negatives.append(torch.from_numpy(negative))
        negatives = torch.stack(negatives, 0)

        return query, positive, negatives, index

    def __len__(self):
        return len(self.poses)


def evaluateResults(seq, global_descs, local_feats, dataset, match_results_save_path=None):
    is_list_style = isinstance(global_descs, (list, tuple))
    if is_list_style:
        if len(global_descs) != 2:
            raise ValueError("当以 list/tuple 形式传入 global_descs 时，期望格式为 [db_descs, q_descs]")
        db_descs = np.asarray(global_descs[0]).astype(np.float32)
        q_descs = np.asarray(global_descs[1]).astype(np.float32)
        desc_dim = db_descs.shape[1]
    else:
        global_descs = np.asarray(global_descs).astype(np.float32)
        desc_dim = global_descs.shape[1]

    gt_thres = 25
    faiss_index = faiss.IndexFlatL2(desc_dim)
    all_positives = 0
    tp = 0
    all_errs = []

    if is_list_style:
        faiss_index.add(db_descs)
        _, predictions = faiss_index.search(q_descs, 1)
    else:
        raise NotImplementedError("非list/tuple模式的db/query划分需根据具体数据集补充db_split_index逻辑")

    # 新增：打印详细信息的标记（选择第20个查询样本作为示例）
    print("\n===== 检索详细信息 =====")
    target_q_idx = 20  # 选择打印第20个查询的详细信息

    for q_idx, pred in enumerate(predictions):
        query_idx = q_idx + dataset.db_split_index
        query_pose = dataset.poses[query_idx]
        if not hasattr(dataset, 'db_split_index'):
            raise AttributeError("数据集对象需包含 db_split_index 属性（标记数据库帧与查询帧的划分）")
        db_poses = dataset.poses[:dataset.db_split_index]

        # 计算正样本（距离小于阈值的DB样本）
        gt_dis = (query_pose - db_poses) ** 2
        positives = np.where(np.sum(gt_dis[:, [3, 7, 11]], axis=1) < gt_thres ** 2)[0]

        # 统计正样本和真阳性
        if len(positives) > 0:
            all_positives += 1
            if pred[0] in positives:
                tp += 1

        # 新增：打印目标查询样本的详细信息
        if q_idx == target_q_idx:
            # 1. 打印Query位姿
            q_x = query_pose[3]  # t0（x平移）
            q_y = query_pose[7]  # t1（y平移）
            q_z = query_pose[11] # t2（z平移）
            print(f"\nQuery {q_idx} - 全局索引: {query_idx} | 位姿(x,y,z): ({q_x:.2f}, {q_y:.2f}, {q_z:.2f})")

            # 2. 打印前5个DB样本的位姿和距离
            print(f"前5个DB样本与Query的距离：")
            for db_i in range(min(5, len(db_poses))):
                db_pose = db_poses[db_i]
                db_x = db_pose[3]
                db_y = db_pose[7]
                db_z = db_pose[11]
                # 计算三维欧氏距离
                dx = q_x - db_x
                dy = q_y - db_y
                dz = q_z - db_z
                dist = np.sqrt(dx**2 + dy**2 + dz**2)
                print(f"DB {db_i} - 位姿(x,y,z): ({db_x:.2f}, {db_y:.2f}, {db_z:.2f}) | 距离: {dist:.2f}m")

            # 3. 打印正样本数量和检索结果
            print(f"Query {q_idx} - 正样本数量（距离<{gt_thres}m）: {len(positives)}")
            print(f"Query {q_idx} - FAISS检索到的DB索引: {pred[0]}")

            # 4. 打印正样本索引范围（前10和后10，避免过长）
            if len(positives) > 0:
                pos_str = ", ".join(map(str, positives[:10]))
                if len(positives) > 10:
                    pos_str += ", ..., " + ", ".join(map(str, positives[-10:]))
                print(f"Query {q_idx} - 正样本索引范围: [{pos_str}]")
            else:
                print(f"Query {q_idx} - 无正样本（距离均≥{gt_thres}m）")

            # 5. 打印检索结果是否为正样本
            is_tp = pred[0] in positives if len(positives) > 0 else False
            print(f"Query {q_idx} - 检索结果是否为正样本: {is_tp}")

        # 位姿误差计算和可视化（保持不变）
        if match_results_save_path is not None:
            if local_feats is None:
                raise ValueError("保存匹配结果时，local_feats（局部特征）不能为None！")
            
            db_idx = pred[0]
            query_im, _ = dataset[query_idx]
            db_im, _ = dataset[db_idx]
            query_im = (query_im.transpose(1, 2, 0) * 256).astype(np.uint8)
            db_im = (db_im.transpose(1, 2, 0) * 256).astype(np.uint8)
            im_side = db_im.shape[0]

            fast = cv2.FastFeatureDetector_create()
            query_kps = fast.detect(query_im, None)
            db_kps = fast.detect(db_im, None)

            H_feat = local_feats.shape[1]
            W_feat = local_feats.shape[2]
            query_des = []
            for kp in query_kps:
                y = int(round(kp.pt[1]))
                x = int(round(kp.pt[0]))
                if 0 <= y < H_feat and 0 <= x < W_feat:
                    query_des.append(local_feats[query_idx][y, x])
            db_des = []
            for kp in db_kps:
                y = int(round(kp.pt[1]))
                x = int(round(kp.pt[0]))
                if 0 <= y < H_feat and 0 <= x < W_feat:
                    db_des.append(local_feats[db_idx][y, x])
            if len(query_des) == 0 or len(db_des) == 0:
                continue
            query_des = np.array(query_des)
            db_des = np.array(db_des)

            matcher = cv2.BFMatcher()
            matches = matcher.knnMatch(query_des, db_des, k=2)
            all_match = [m[0] for m in matches]
            if len(all_match) == 0:
                continue

            points1 = (np.array([[im_side//2, im_side//2]]) - np.float32([kp.pt for kp in query_kps])) * 0.4
            points2 = (np.array([[im_side//2, im_side//2]]) - np.float32([kp.pt for kp in db_kps])) * 0.4
            points1 = points1[[m.queryIdx for m in all_match]]
            points2 = points2[[m.trainIdx for m in all_match]]

            try:
                H, mask, max_csc_num = rigidRansac(points1, points2)
            except Exception as e:
                print(f"RANSAC计算失败（query_idx={query_idx}）: {str(e)}")
                continue

            q_pose_3d = query_pose[:12].reshape(3, 4)
            db_pose_3d = dataset.poses[db_idx][:12].reshape(3, 4)
            q_pose_2d = np.hstack([q_pose_3d[:2, :2], q_pose_3d[:2, 3].reshape(-1, 1)])
            db_pose_2d = np.hstack([db_pose_3d[:2, :2], db_pose_3d[:2, 3].reshape(-1, 1)])
            q_pose_2d = np.vstack([q_pose_2d, [0, 0, 1]])
            db_pose_2d = np.vstack([db_pose_2d, [0, 0, 1]])

            relative_gt = np.linalg.inv(db_pose_2d).dot(q_pose_2d)
            relative_H = np.vstack([H, [0, 0, 1]])
            err_matrix = np.linalg.inv(relative_H).dot(relative_gt)
            err_theta = np.abs(np.arctan2(err_matrix[0, 1], err_matrix[0, 0]) / np.pi * 180)
            err_trans = np.sqrt(err_matrix[0, 2] ** 2 + err_matrix[1, 2] ** 2)
            all_errs.append([err_trans, err_theta])

            good_match = [all_match[i] for i in range(len(mask)) if mask[i]]
            db_im_marked = cv2.convertScaleAbs(db_im * 3)
            db_im_marked[:, :, :2] = 0

            match_im = cv2.drawMatches(
                query_im, query_kps, db_im_marked, db_kps, good_match,
                None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )

            out_im_h = match_im.shape[0] * 2
            out_im_w = db_im.shape[1] * 3
            out_im = np.zeros((out_im_h, out_im_w, 3), dtype=np.uint8)
            out_im[:match_im.shape[0], :db_im.shape[1]] = query_im
            out_im[:match_im.shape[0], db_im.shape[1]:db_im.shape[1]*2] = db_im_marked
            out_im[:match_im.shape[0], db_im.shape[1]*2:] = cv2.addWeighted(query_im, 0.5, db_im_marked, 0.5, 0)
            out_im[-match_im.shape[0]:, :db_im.shape[1]*2] = match_im

            rot_angle = np.arctan2(-H[0, 1], H[0, 0]) / np.pi * 180
            rot_mat = cv2.getRotationMatrix2D((im_side//2, im_side//2), rot_angle, 1.0)
            rot_mat[0, 2] -= H[1, 2] / 0.4
            rot_mat[1, 2] -= H[0, 2] / 0.4
            im_warp = cv2.warpAffine(db_im_marked, rot_mat, query_im.shape[:2])
            out_im[-match_im.shape[0]:, db_im.shape[1]*2:] = cv2.addWeighted(query_im, 0.5, im_warp, 0.5, 0)

            save_name = f"{1000000 + query_idx:06d}.png"
            cv2.imwrite(join(match_results_save_path, save_name), out_im)

    # 打印最终召回率
    recall_top1 = tp / all_positives if all_positives > 0 else 0.0
    print(f"\n===== 评估结果 =====")
    print(f"Recall@1: {recall_top1:.4f} ({recall_top1*100:.2f}%)")

    if match_results_save_path is not None and len(all_errs) > 0:
        all_errs = np.array(all_errs)
        success_mask = (all_errs[:, 0] < 2) & (all_errs[:, 1] < 5)
        success_rate = np.sum(success_mask) / all_positives if all_positives > 0 else 0.0
        mean_trans_err = np.mean(all_errs[success_mask, 0]) if success_mask.any() else 0.0
        mean_rot_err = np.mean(all_errs[success_mask, 1]) if success_mask.any() else 0.0
        return recall_top1, success_rate, mean_trans_err, mean_rot_err
    else:
        return recall_top1, 0.0, 0.0, 0.0


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return None, None, None, None
    
    query, positive, negatives, indices = zip(*batch)
    query = torch.tensor(np.array(query), dtype=torch.float32)
    positive = torch.tensor(np.array(positive), dtype=torch.float32)
    negatives = torch.cat(negatives, dim=0)
    indices = list(indices)
    
    return query, positive, negatives, indices