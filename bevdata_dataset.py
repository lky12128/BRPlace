import os
from os.path import join, exists, splitext
import numpy as np
import cv2
import torch
import torch.utils.data as data
import h5py
import faiss
from RANSAC import rigidRansac  


bevdata_train_seq =   {"query": ['bc_20230920_1','if_20231213_4'],
                        "db": ["if_20231208_4",'bc_20231105_6']}
bevdata_val_seq =     {"query": ['bc_20230920_1','if_20231213_4'],
                        "db": ["if_20231208_4",'bc_20231105_6']}
bevdata_test_seq =     {"query": ['81r_20240116_2','81r_20240116_eve_3'],
                        "db": ["81r_20240123_2",'iaef_20240115_2']}
bevdata_train_pairs = [("ss_20231109_4", "ss_20231105_aft_5"),
                       ("if_20240116_5",  "if_20231208_4"),
                       ("sl_20231105_2",  "sl_20231105_aft_4")]
bevdata_train_split=1000
db_split_index=300   #？？？


class InferDataset(data.Dataset):
    """推理数据集（用于测试和验证阶段）"""
    def __init__(self, seq, dataset_path='./datasets/bevdata', 
                 data_type='lidar_bev_z', sample_inteval=1):
        super().__init__()
        
        self.dataset_path = dataset_path  
        self.data_type = data_type  
        self.datapath_type = dataset_path + '/' + data_type + '/' + 'val'  # 验证集路径
        self.seq = seq
        self.db_split_index = db_split_index  # 新增：DB与Query的划分索引
        self.sample_inteval = sample_inteval  # 帧采样间隔

        # 读取图像路径
        img_dir = join(self.datapath_type, seq, 'images') 
        if not exists(img_dir):
            raise FileNotFoundError(f"图像目录不存在: {img_dir}")
        imgs_p = os.listdir(img_dir)
        imgs_p = sorted(imgs_p)  # 按文件名排序
        # 按帧号排序并采样
        # imgs_p = sorted(imgs_p, key=lambda x: int(splitext(x)[0]))
        self.imgs_path = [join(img_dir, imgs_p[i]) for i in range(0, len(imgs_p), sample_inteval)]
        
        # self.pose_dir = join(self.datapath_type, seq, 'poses')
        # self.poses = self._load_poses()
        

        #加载位姿（假设poses.txt中每行对应一个位姿）
        self.poses = np.loadtxt(
            join(self.datapath_type, seq, 'poses', 'poses.txt'),
            dtype=np.float32
        )[::self.sample_inteval]  # 按采样间隔提取位姿

        #确保位姿数量与图像数量一致
        if len(self.poses) != len(self.imgs_path):
            min_len = min(len(self.poses), len(self.imgs_path))
            self.poses = self.poses[:min_len]
            self.imgs_path = self.imgs_path[:min_len]
            print(f"警告：位姿与图像数量不匹配，已截断至 {min_len} 个样本")
            
    def _load_poses(self, num=0):
        """加载位姿文件（每个图像对应一个位姿文件）"""
        poses = []
        for img_name in self.imgs_path:
            img_basename = splitext(os.path.basename(img_name))[0]
            pose_path = join(self.pose_dir, f"{img_basename}.txt")

            if not exists(pose_path):
                raise FileNotFoundError(f"位姿文件缺失: {pose_path}")
            
            try:
                pose = np.loadtxt(pose_path, dtype=np.float32)
                if pose.shape != (12,):
                    raise ValueError(f"位姿文件格式错误（需12个数据）: {pose_path}")
                poses.append(pose)
                if num > 0 and len(poses) >= num:
                    break
            except Exception as e:
                raise RuntimeError(f"读取位姿文件失败: {str(e)}")
        
        poses_np = np.array(poses)
        if num > 0 and len(poses_np) > num:
            poses_np = poses_np[:num]
        
        # 确保位姿与图像数量一致
        if len(poses_np) != len(self.imgs_path[:len(poses_np)]):
            raise ValueError(f"位姿数量（{len(poses_np)}）与图像数量不匹配！")
                # 在循环读取位姿时添加

        return poses_np

    def __getitem__(self, index):
        img_path = self.imgs_path[index]
        img = cv2.imread(img_path, 0)  # 读取灰度图

        if img is None:
            raise FileNotFoundError(f"无法读取图像（文件损坏或路径错误）: {img_path}")
        img = (img.astype(np.float32)) / 255.0  # 归一化
        # 转换为3通道（适配模型输入）
        img = img[np.newaxis, :, :].repeat(3, 0)  # 形状：[3, H, W]
        return img, index
    
    def __len__(self):
        return len(self.imgs_path)


def evaluateResults(seq, global_descs, local_feats, dataset, match_results_save_path=None):
    """
    评估BEV检索结果，计算Recall@1、成功率、平均平移误差和平均旋转误差
    
    参数:
        seq: 序列名称
        global_descs: 全局描述符，格式为[db_descs, q_descs]或单序列特征
        local_feats: 局部特征（用于可视化匹配）
        dataset: 数据集对象，需包含poses和db_split_index属性
        match_results_save_path: 匹配结果可视化保存路径（None表示不保存）
    
    返回:
        recall_top1: Top-1召回率
        success_rate: 位姿估计成功率（平移<2m且旋转<5°）
        mean_trans_err: 成功匹配的平均平移误差
        mean_rot_err: 成功匹配的平均旋转误差
    """
    # 区分输入格式：列表形式[db_descs, q_descs]或单序列全局特征
    is_list_style = isinstance(global_descs, (list, tuple))
    if is_list_style:
        if len(global_descs) != 2:
            raise ValueError("当以list/tuple传入global_descs时，需为[db_descs, q_descs]")
        db_descs = np.asarray(global_descs[0]).astype(np.float32)
        q_descs = np.asarray(global_descs[1]).astype(np.float32)
        desc_dim = db_descs.shape[1]
    else:
        # 单序列情况：通过db_split_index拆分DB和Query
        global_descs = np.asarray(global_descs).astype(np.float32)
        desc_dim = global_descs.shape[1]
        db_idx = dataset.db_split_index
        db_descs = global_descs[:db_idx].astype(np.float32)
        q_descs = global_descs[db_idx:].astype(np.float32)

    # 地面真值距离阈值（平移误差<25米视为正样本）
    gt_thres = 25
    # 初始化FAISS索引（L2距离）
    faiss_index = faiss.IndexFlatL2(desc_dim)
    faiss_index.add(db_descs)
    _, predictions = faiss_index.search(q_descs, 1)  # 检索Top-1结果

    # 评估指标初始化
    all_positives = 0  # 总正样本数（存在符合阈值的地面真值）
    tp = 0             # 真阳性数（检索结果在正样本内）
    all_errs = []      # 存储位姿误差（仅在保存结果时使用）

    # 遍历所有查询样本
    for q_idx, pred in enumerate(predictions):
        # 当前查询样本的全局索引（Query部分 = db_split_index + 本地索引）
        query_global_idx = q_idx + dataset.db_split_index
        query_pose = dataset.poses[query_global_idx]
        db_poses = dataset.poses[:dataset.db_split_index]  # 所有DB样本的位姿

        # 计算查询与所有DB样本的平移误差（x/y/z轴，对应位姿矩阵第4/8/12列）
        gt_dis = (query_pose - db_poses) ** 2
        positives = np.where(np.sum(gt_dis[:, [3, 7, 11]], axis=1) < gt_thres **2)[0]

        # 统计正样本和真阳性
        if len(positives) > 0:
            all_positives += 1
            if pred[0] in positives:
                tp += 1

        # 在循环处理每个Query时添加（以q_idx=10为例）
        if q_idx == 20:  # 仅打印第10个Query的结果，避免日志过多
            print(f"\nQuery {q_idx} - 位姿x: {query_pose[3]}, y: {query_pose[7]}")
            # 打印前5个DB样本的位姿x/y和距离
            for db_i in range(5):
                db_x = db_poses[db_i][3]
                db_y = db_poses[db_i][7]
                db_z = db_poses[db_i][11]
                dx = query_pose[3] - db_x
                dy = query_pose[7] - db_y
                dz = query_pose[11] - db_z
                dist = np.sqrt(dx**2 + dy**2 + dz**2)
                print(f"DB {db_i} - x: {db_x}, y: {db_y}, z: {db_z}, 距离: {dist:.2f}m")
            # 打印当前Query的正样本数量
            print(f"Query {q_idx} - 正样本数量（距离<5m）: {len(positives)}")
            print(f"Query {q_idx} - FAISS检索到的DB索引: {pred[0]}")
            print(f"Query {q_idx} - 正样本索引范围: {positives[:]}...")  # 打印前100个正样本索引
    # 检查检索结果是否在正样本内
            is_tp = pred[0] in positives
            print(f"Query {q_idx} - 检索结果是否为正样本: {is_tp}")
        # 位姿误差计算和可视化（仅当需要保存结果时）
        if match_results_save_path is not None:
            # 检索到的DB样本索引
            db_idx = pred[0]
            if db_idx < 0 or db_idx >= len(db_poses):
                continue  # 无效索引跳过

            # 1. 读取并预处理图像
            query_im, _ = dataset[query_global_idx]
            db_im, _ = dataset[db_idx]
            # 从CHW转HWC，像素值还原为[0,255]
            query_im = (query_im.transpose(1, 2, 0) * 256).astype(np.uint8)
            db_im = (db_im.transpose(1, 2, 0) * 256).astype(np.uint8)
            im_side = db_im.shape[0]  # 图像边长（假设正方形）

            # 2. FAST特征点检测
            fast = cv2.FastFeatureDetector_create()
            query_kps = fast.detect(query_im, None)
            db_kps = fast.detect(db_im, None)

            # 3. 提取局部特征（过滤越界关键点）
            H_feat = local_feats.shape[1] if local_feats is not None else im_side
            W_feat = local_feats.shape[2] if local_feats is not None else im_side

            query_des, valid_q_kps = [], []
            for kp in query_kps:
                y, x = int(round(kp.pt[1])), int(round(kp.pt[0]))
                if 0 <= y < H_feat and 0 <= x < W_feat and local_feats is not None:
                    query_des.append(local_feats[query_global_idx][y, x])
                    valid_q_kps.append(kp)

            db_des, valid_db_kps = [], []
            for kp in db_kps:
                y, x = int(round(kp.pt[1])), int(round(kp.pt[0]))
                if 0 <= y < H_feat and 0 <= x < W_feat and local_feats is not None:
                    db_des.append(local_feats[db_idx][y, x])
                    valid_db_kps.append(kp)

            # 无有效特征点时跳过
            if len(query_des) == 0 or len(db_des) == 0:
                continue
            query_des = np.array(query_des)
            db_des = np.array(db_des)

            # 4. 特征匹配与RANSAC
            matcher = cv2.BFMatcher()
            matches = matcher.knnMatch(query_des, db_des, k=2)
            all_match = [m[0] for m in matches]  # 取Top-1匹配

            # 坐标中心化与缩放
            points1 = (np.array([[im_side//2, im_side//2]]) - 
                      np.float32([kp.pt for kp in valid_q_kps])) * 0.4
            points2 = (np.array([[im_side//2, im_side//2]]) - 
                      np.float32([kp.pt for kp in valid_db_kps])) * 0.4
            points1 = points1[[m.queryIdx for m in all_match]]
            points2 = points2[[m.trainIdx for m in all_match]]

            try:
                # 估计刚性变换
                H, mask, max_csc_num = rigidRansac(points1, points2)
            except Exception as e:
                print(f"RANSAC计算失败（query_idx={query_global_idx}）: {e}")
                continue

            # 5. 计算位姿误差
            # 位姿格式：[R00, R01, R02, t0, R10, R11, R12, t1, R20, R21, R22, t2]
            q_pose_3d = dataset.poses[query_global_idx][:12].reshape(3, 4)
            db_pose_3d = dataset.poses[db_idx][:12].reshape(3, 4)

            # 构建2D位姿矩阵（旋转+平移）
            q_pose_2d = np.hstack([q_pose_3d[:2, :2], q_pose_3d[:2, 3].reshape(-1, 1)])
            db_pose_2d = np.hstack([db_pose_3d[:2, :2], db_pose_3d[:2, 3].reshape(-1, 1)])
            q_pose_2d = np.vstack([q_pose_2d, [0, 0, 1]])  # 补全为3x3
            db_pose_2d = np.vstack([db_pose_2d, [0, 0, 1]])

            # 计算相对位姿与误差
            relative_gt = np.linalg.inv(db_pose_2d).dot(q_pose_2d)
            relative_H = np.vstack([H, [0, 0, 1]])
            err_matrix = np.linalg.inv(relative_H).dot(relative_gt)

            err_theta = np.abs(np.arctan2(err_matrix[0, 1], err_matrix[0, 0]) / np.pi * 180)
            err_trans = np.sqrt(err_matrix[0, 2]** 2 + err_matrix[1, 2] **2)
            all_errs.append([err_trans, err_theta])

            # 6. 可视化匹配结果
            good_match = [all_match[i] for i in range(len(mask)) if mask[i]]
            # 标记DB图像为红色
            db_im_marked = cv2.convertScaleAbs(db_im * 3)
            db_im_marked[:, :, :2] = 0  # 关闭蓝、绿色通道

            # 绘制匹配对
            match_im = cv2.drawMatches(
                query_im, valid_q_kps, db_im_marked, valid_db_kps, good_match,
                None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )

            # 构建输出画布
            out_im_h = match_im.shape[0] * 2
            out_im_w = db_im.shape[1] * 3
            out_im = np.zeros((out_im_h, out_im_w, 3), dtype=np.uint8)

            # 上半部分：原始图、标记图、融合图
            out_im[:match_im.shape[0], :db_im.shape[1]] = query_im
            out_im[:match_im.shape[0], db_im.shape[1]:db_im.shape[1]*2] = db_im_marked
            out_im[:match_im.shape[0], db_im.shape[1]*2:] = cv2.addWeighted(
                query_im, 0.5, db_im_marked, 0.5, 0
            )

            # 下半部分左：匹配对可视化
            out_im[-match_im.shape[0]:, :db_im.shape[1]*2] = match_im

            # 下半部分右：配准结果
            rot_angle = np.arctan2(-H[0, 1], H[0, 0]) / np.pi * 180
            rot_mat = cv2.getRotationMatrix2D((im_side//2, im_side//2), rot_angle, 1.0)
            rot_mat[0, 2] -= H[1, 2] / 0.4
            rot_mat[1, 2] -= H[0, 2] / 0.4
            im_warp = cv2.warpAffine(db_im_marked, rot_mat, query_im.shape[:2])
            out_im[-match_im.shape[0]:, db_im.shape[1]*2:] = cv2.addWeighted(
                query_im, 0.5, im_warp, 0.5, 0
            )

            # 保存可视化结果
            if not exists(match_results_save_path):
                os.makedirs(match_results_save_path)
            save_name = f"{1000000 + query_global_idx:06d}.png"
            cv2.imwrite(join(match_results_save_path, save_name), out_im)

    # 计算Top-1召回率（避免除零错误）
    recall_top1 = tp / all_positives if all_positives > 0 else 0.0

    # 计算位姿误差指标
    if match_results_save_path is not None and len(all_errs) > 0:
        all_errs = np.array(all_errs)
        success_mask = (all_errs[:, 0] < 2) & (all_errs[:, 1] < 5)
        success_rate = np.sum(success_mask) / all_positives if all_positives > 0 else 0.0
        mean_trans_err = np.mean(all_errs[success_mask, 0]) if success_mask.any() else 0.0
        mean_rot_err = np.mean(all_errs[success_mask, 1]) if success_mask.any() else 0.0
    else:
        # 当不保存结果时，返回默认值
        success_rate = 0.0
        mean_trans_err = 0.0
        mean_rot_err = 0.0

    # 始终返回四个值，确保一致性
    return recall_top1, success_rate, mean_trans_err, mean_rot_err


class TrainingDataset(data.Dataset):
    """训练数据集（用于训练阶段，支持硬样本挖掘）"""
    def __init__(self, dataset_path='./datasets/bevdata', data_type='', seq=''):
        super().__init__()
        self.dataset_path = dataset_path
        self.seq = seq
        self.data_type = data_type
        self.datapath_type = dataset_path + '/' + data_type + '/' + 'train'  # 训练集路径

        # 读取图像路径
        img_dir = join(self.datapath_type, seq, 'images')
        if not exists(img_dir):
            raise FileNotFoundError(f"图像目录不存在: {img_dir}")
        imgs_p = os.listdir(img_dir)
        imgs_p = sorted(imgs_p)  # 按文件名排序
        # imgs_p = sorted(imgs_p, key=lambda x: int(splitext(x)[0]))  # 按帧号排序
        self.imgs_path = [join(img_dir, img) for img in imgs_p]

        # 加载前3000帧位姿
        # self.pose_dir = join(self.datapath_type, seq, 'poses')
        # self.poses = self._load_poses(3000)
        
# ==================== 修改开始 ====================
        # 加载位姿（从单一 poses.txt 文件，逻辑同 InferDataset）
        
        # 1. 定义 poses.txt 文件路径
        pose_file_path = join(self.datapath_type, seq, 'poses', 'poses.txt')
        if not exists(pose_file_path):
             raise FileNotFoundError(f"训练位姿文件缺失: {pose_file_path}")
        
        # 2. 从 poses.txt 加载所有位姿
        try:
            all_poses = np.loadtxt(pose_file_path, dtype=np.float32)
        except Exception as e:
            raise RuntimeError(f"读取训练位姿文件失败: {str(e)}")

        # 3. 限制加载的帧数（与原始 _load_poses(3000) 逻辑保持一致）
        num_to_load = 3000
        if len(all_poses) > num_to_load:
            self.poses = all_poses[:num_to_load]
        else:
            self.poses = all_poses

        # 4. 确保图像列表 (self.imgs_path) 与位姿列表 (self.poses) 长度一致
        #    （这是最关键的一步，防止数据错配）
        min_len = min(len(self.poses), len(self.imgs_path))
        if len(self.poses) != len(self.imgs_path):
            print(f"警告：训练集位姿与图像数量不匹配（{len(self.poses)} vs {len(self.imgs_path)}），"
                  f"已截断至 {min_len} 个样本")
            self.poses = self.poses[:min_len]
            self.imgs_path = self.imgs_path[:min_len]
        
        # 正负样本阈值与数量设置
        self.pos_thres = 10  # 正样本：平移误差<10米
        self.neg_thres = 50  # 负样本：平移误差>50米
        self.num_neg = 10    # 每个Query的负样本数
        self.positives = []  # 正样本索引列表
        self.negatives = []  # 负样本索引列表

        # 预计算正负样本
        for qi in range(len(self.poses)):
            q_pose = self.poses[qi]
            # 计算与所有样本的距离
            dises = np.sqrt(np.sum(((q_pose - self.poses)**2)[:, [3,7,11]], axis=1))
            indexes = np.argsort(dises)
            
            # 筛选正样本（排除自身）
            pos_indexes = indexes[dises[indexes] < self.pos_thres]
            pos_indexes = pos_indexes[pos_indexes != qi]
            self.positives.append(pos_indexes)
            
            # 筛选负样本
            neg_indexes = indexes[dises[indexes] > self.neg_thres]
            self.negatives.append(neg_indexes)

        self.mining = False  # 是否启用硬样本挖掘
        self.cache = None    # HDF5特征缓存路径

    def _load_poses(self, num=0):
        """加载位姿文件（每个图像对应一个位姿文件）"""
        poses = []
        for img_name in self.imgs_path:
            img_basename = splitext(os.path.basename(img_name))[0]
            pose_path = join(self.pose_dir, f"{img_basename}.txt")
            
            if not exists(pose_path):
                raise FileNotFoundError(f"位姿文件缺失: {pose_path}")
            
            try:
                pose = np.loadtxt(pose_path, dtype=np.float32)
                if pose.shape != (12,):
                    raise ValueError(f"位姿文件格式错误（需12个数据）: {pose_path}")
                poses.append(pose)
                
                if num > 0 and len(poses) >= num:
                    break
            except Exception as e:
                raise RuntimeError(f"读取位姿文件失败: {str(e)}")
        
        poses_np = np.array(poses)
        if num > 0 and len(poses_np) > num:
            poses_np = poses_np[:num]
        
        # 确保位姿与图像数量一致
        if len(poses_np) != len(self.imgs_path[:len(poses_np)]):
            raise ValueError(f"位姿数量（{len(poses_np)}）与图像数量不匹配！")
        
        return poses_np

    def refreshCache(self):
        """加载HDF5特征用于硬样本挖掘"""
        if self.cache is None:
            raise ValueError("请先设置self.cache为HDF5特征文件路径！")
        with h5py.File(self.cache, 'r') as h5:
            self.h5feat = np.array(h5.get("features"))
            if self.h5feat.shape[0] != len(self.poses):
                raise ValueError(f"HDF5特征数量与数据集数量不匹配！")

    def __getitem__(self, index):
        """获取三元组（Query, Positive, Negatives）"""
        if self.mining and hasattr(self, 'h5feat'):
            # 硬样本挖掘：选择最难分的正样本和负样本
            pos_indexes = self.positives[index]
            if len(pos_indexes) == 0:
                raise ValueError(f"帧 {index} 无正样本！")
            q_feat = self.h5feat[index]
            
            # 最难分正样本（距离最远）
            pos_feats = self.h5feat[pos_indexes]
            pos_dist = np.sqrt(np.sum((q_feat - pos_feats)** 2, axis=1))
            pos_idx = pos_indexes[np.argmax(pos_dist)]
            
            # 最难分负样本（距离最近）
            neg_indexes = self.negatives[index]
            if len(neg_indexes) < self.num_neg:
                raise ValueError(f"帧 {index} 负样本不足！")
            neg_feats = self.h5feat[neg_indexes]
            neg_dist = np.sqrt(np.sum((q_feat - neg_feats)**2, axis=1))
            neg_idx = neg_indexes[np.argsort(neg_dist)[:self.num_neg]]
        else:
            # 普通模式：随机选择正负样本
            pos_indexes = self.positives[index]
            if len(pos_indexes) == 0:
                raise ValueError(f"帧 {index} 无正样本！")
            pos_idx = np.random.choice(pos_indexes, 1)[0]
            
            neg_indexes = self.negatives[index]
            if len(neg_indexes) < self.num_neg:
                raise ValueError(f"帧 {index} 负样本不足！")
            neg_idx = np.random.choice(neg_indexes, self.num_neg, replace=False)

        # 读取并预处理Query图像
        query_path = self.imgs_path[index]
        query = cv2.imread(query_path, 0)
        if query is None:
            raise FileNotFoundError(f"无法读取Query图像: {query_path}")
        # 随机旋转增强
        rot_mat = cv2.getRotationMatrix2D((query.shape[1]//2, query.shape[0]//2), 
                                         np.random.randint(0, 360), 1.0)
        query = cv2.warpAffine(query, rot_mat, query.shape[:2])
        query = query.astype(np.float32)[np.newaxis, :, :].repeat(3, 0)

        # 读取并预处理正样本图像
        pos_path = self.imgs_path[int(pos_idx)]
        positive = cv2.imread(pos_path, 0)
        if positive is None:
            raise FileNotFoundError(f"无法读取正样本图像: {pos_path}")
        rot_mat = cv2.getRotationMatrix2D((positive.shape[1]//2, positive.shape[0]//2), 
                                         np.random.randint(0, 360), 1.0)
        positive = cv2.warpAffine(positive, rot_mat, positive.shape[:2])
        positive = positive.astype(np.float32)[np.newaxis, :, :].repeat(3, 0)

        # 读取并预处理负样本图像
        negatives = []
        for ni in neg_idx:
            neg_path = self.imgs_path[int(ni)]
            negative = cv2.imread(neg_path, 0)
            if negative is None:
                raise FileNotFoundError(f"无法读取负样本图像: {neg_path}")
            rot_mat = cv2.getRotationMatrix2D((negative.shape[1]//2, negative.shape[0]//2), 
                                             np.random.randint(0, 360), 1.0)
            negative = cv2.warpAffine(negative, rot_mat, negative.shape[:2])
            negative = negative.astype(np.float32)[np.newaxis, :, :].repeat(3, 0)
            negatives.append(torch.from_numpy(negative))
        negatives = torch.stack(negatives, 0)

        return query, positive, negatives, index

    def __len__(self):
        return len(self.poses)


def collate_fn(batch):
    """自定义批处理函数（适配负样本堆叠）"""
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return None, None, None, None
    
    query, positive, negatives, indices = zip(*batch)
    # 转Tensor并拼接
    query = torch.tensor(np.array(query), dtype=torch.float32)
    positive = torch.tensor(np.array(positive), dtype=torch.float32)
    negatives = torch.cat(negatives, dim=0)  # 负样本拼接为[B*num_neg, C, H, W]
    indices = list(indices)
    
    return query, positive, negatives, indices
