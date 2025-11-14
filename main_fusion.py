import argparse
from math import ceil
import random
import shutil
import json
from os.path import join, exists, isfile
from os import makedirs
import os
from datetime import datetime
from torch.cuda.amp import GradScaler, autocast

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import h5py
from tensorboardX import SummaryWriter
import numpy as np
from tqdm import tqdm

# --- 导入新的模型和数据集 ---
from fusion_model import FusionPlaceModel
import fusion_dataset as ds_module # 导入新的数据集文件


def get_args():
    parser = argparse.ArgumentParser(description='FusionPlace (BEV+Range) V6.5 (Decoupled)')
    
    parser.add_argument('--mode', type=str, default='test', help='Mode', choices=['train', 'test'])
    
    # 路径参数
    parser.add_argument('--bev_data_path', type=str, 
                        default='./datasets/snail/radar_bev_z', 
                        help='BEV 图像 和 Poses 的根路径 (e.g., .../radar_bev_z)')
    parser.add_argument('--range_data_path', type=str, 
                        default='/mnt/kaiyan/datasets/snail_radar/', 
                        help='NPY 点云的根路径 (e.g., .../snail_radar)')
    parser.add_argument('--range_data_type', type=str, 
                        default='eagleg7_npy', 
                        help='Range NPY点云的子目录名 (e.g., eagleg7_npy)')
    
    # 验证集DB与Query序列 (从 bevdata_dataset 导入)
    parser.add_argument('--val_db_seq', type=str, default="20231201_2", help='测试时用作数据库的序列')
    parser.add_argument('--val_q_seq', type=str, default="20231201_3", help='测试时用作查询的序列')
    # ================================================

    # 通用训练参数
    parser.add_argument('--batchSize', type=int, default=1,
                        help='训练批量')
    parser.add_argument('--cacheBatchSize', type=int, default=1, 
                        help='缓存/测试时的批量')
    parser.add_argument('--nEpochs', type=int, default=40, help='训练轮数')
    parser.add_argument('--threads', type=int, default=0, help='数据加载线程数')
    parser.add_argument('--seed', type=int, default=1024, help='随机种子')
    
    # 优化器
    parser.add_argument('--lr', type=float, default=0.0001, help='学习率')
    parser.add_argument('--lrStep', type=float, default=10, help='学习率衰减步长')
    parser.add_argument('--lrGamma', type=float, default=0.5, help='学习率衰减系数')
    parser.add_argument('--weightDecay', type=float, default=0.001, help='权重衰减')
    
    # 权重和日志
    parser.add_argument('--runsPath', type=str, default='./runs/', help='训练日志保存路径')
    parser.add_argument('--cachePath', type=str, default='./cache/fusion', help='特征缓存保存路径')
    parser.add_argument('--bev_weights_path', type=str, 
                        default='runs/bevdata_Nov04_00-26-28/model_best.pth.tar', # (BEV baseline 权重)
                        help='BEVPlace++ (REIN) 预训练模型路径')
    parser.add_argument('--diffloc_weights_path', type=str, 
                        default='./runs/DINO/nclt.pth', # (DINO nclt 权重)
                        help='DiffLoc/DINO (Range Extractor) 预训练模型路径')
    parser.add_argument('--range_dim', type=int, default=384, 
                        help='Range 特征维度 (384 for ViT-S, 768 for ViT-B)')
    
    parser.add_argument('--sample_interval', type=int, default=10, 
                        help='推理时的帧采样间隔')
    parser.add_argument('--match_save_path', type=str, 
                        default='./runs/test_matches/', 
                        help='测试时保存匹配结果的路径')

    opt = parser.parse_args()
    return opt

class TripletLoss(nn.Module):
    def __init__(self):
        super(TripletLoss, self).__init__()
        self.margin = 0.3
    def forward(self, anchor, positive, negative):
        pos_dist = torch.sqrt((anchor - positive).pow(2).sum())
        neg_dist = torch.sqrt((anchor - negative).pow(2).sum(1))
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss

from torch.cuda.amp import GradScaler, autocast
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import h5py
from tqdm import tqdm
import numpy as np


def train_epoch(epoch, model, train_set, opt, device, writer, optimizer):
    """
    完整的训练轮次实现，包含：
    1. 硬样本挖掘特征缓存构建
    2. 混合精度训练（解决内存不足）
    3. 内存优化（定期清理缓存）
    4. 损失计算与日志记录
    """
    epoch_loss = 0.0
    n_batches = (len(train_set) + opt.batchSize - 1) // opt.batchSize  # 总批次数
    criterion = TripletLoss().to(device)
    scaler = GradScaler()  # 混合精度缩放器

    # --------------------------
    # 步骤1：构建硬样本挖掘特征缓存
    # --------------------------
    if epoch >= 0:  # 每轮都更新缓存（确保特征与当前模型状态一致）
        print(f'====> Epoch {epoch}: 构建硬样本挖掘特征缓存')
        train_set.mining = False  # 缓存阶段不启用硬样本挖掘
        
        # 确保缓存路径存在
        if not exists(opt.cachePath):
            makedirs(opt.cachePath)
        train_set.cache = join(opt.cachePath, 'desc_cen_hardmining_fusion.hdf5')
        
        # 写入特征缓存（使用推理模式，不计算梯度）
        with h5py.File(train_set.cache, mode='w') as h5:
            pool_size = model.global_feat_dim  # 全局特征维度（如8192）
            h5feat = h5.create_dataset(
                "features", 
                [len(train_set), pool_size],  # 形状：[样本数, 特征维度]
                dtype=np.float32
            )
            
            # 缓存加载器（使用推理专用collate_fn）
            train_loader_cache = DataLoader(
                dataset=train_set,
                num_workers=opt.threads,
                batch_size=opt.cacheBatchSize,
                shuffle=False,
                collate_fn=collate_fn_inference  # 处理推理阶段的数据拼接
            )
            
            model.eval()  # 推理模式（关闭 dropout/bn更新）
            with torch.no_grad():
                for iteration, (query_data, indices) in enumerate(
                    tqdm(train_loader_cache, desc="构建缓存"), 1
                ):
                    # 跳过空批次
                    if query_data is None:
                        continue
                    (query_bev, query_range) = query_data  # 解压数据
                    
                    # 移动到设备并计算特征
                    query_bev = query_bev.to(device, non_blocking=True)
                    query_range = query_range.to(device, non_blocking=True)
                    
                    # 混合精度推理（进一步减少缓存阶段内存占用）
                    with autocast():
                        global_descs = model(query_bev, query_range)
                    
                    # 写入HDF5
                    h5feat[indices, :] = global_descs.detach().cpu().numpy()
            model.train()  # 恢复训练模式

    # --------------------------
    # 步骤2：准备训练数据
    # --------------------------
    train_set.mining = True  # 启用硬样本挖掘
    train_set.refreshCache()  # 加载缓存的特征用于硬样本选择
    
    # 训练数据加载器（使用训练专用collate_fn）
    train_loader = DataLoader(
        dataset=train_set,
        num_workers=opt.threads,
        batch_size=opt.batchSize,
        shuffle=True,
        pin_memory=True,  # 加速CPU到GPU的数据传输
        collate_fn=ds_module.collate_fn  # 处理训练阶段的三元组数据
    )

    # --------------------------
    # 步骤3：训练循环
    # --------------------------
    model.train()  # 确保模型处于训练模式
    for iteration, batch_data in enumerate(
        tqdm(train_loader, desc=f"Epoch {epoch} 训练"), 1
    ):
        # 清理GPU缓存（减少内存碎片）
        torch.cuda.empty_cache()
        
        # 跳过空批次
        if batch_data[0] is None:
            print(f"警告: 训练批次 {iteration} 为空，已跳过")
            continue
        
        # 解压批次数据（查询/正样本/负样本/索引）
        (query_data, positive_data, negative_data, indices) = batch_data
        query_bev, query_range = query_data
        positive_bev, positive_range = positive_data
        negatives_bev, negatives_range = negative_data
        
        # 获取批次大小（用于后续特征分割）
        B = query_bev.shape[0]
        
        # 合并输入并移动到GPU（减少设备传输次数）
        bev_inputs = torch.cat([query_bev, positive_bev, negatives_bev]).to(
            device, non_blocking=True
        )
        range_inputs = torch.cat([query_range, positive_range, negatives_range]).to(
            device, non_blocking=True
        )
        
        # 清零梯度
        optimizer.zero_grad(set_to_none=True)  # 更高效的梯度清零

        # --------------------------
        # 混合精度前向传播
        # --------------------------
        with autocast():
            # 计算所有输入的全局特征
            global_descs = model(bev_inputs, range_inputs)
            
            # 分割特征：查询（Q）、正样本（P）、负样本（N）
            global_descs_Q, global_descs_P, global_descs_N = torch.split(
                global_descs, 
                [B, B, negatives_bev.shape[0]]  # 负样本数量 = 总负样本数
            )
            
            # 计算三元组损失（硬负样本策略）
            loss = 0.0
            num_negs_per_q = negatives_bev.shape[0] // B  # 每个查询的负样本数
            for i in range(B):
                # 对每个查询，计算与正样本的距离和最硬负样本的距离
                q_feat = global_descs_Q[i]
                p_feat = global_descs_P[i]
                n_feats = global_descs_N[num_negs_per_q*i : num_negs_per_q*(i+1)]
                
                # 计算损失并取最大（ hardest negative mining ）
                # 修改（）
                avg_loss = torch.mean(criterion(q_feat, p_feat, n_feats)) # <--- 从 max 改成 mean
                loss += avg_loss
                            
            # 平均到批次大小
            loss = loss / B

        # --------------------------
        # 混合精度反向传播
        # --------------------------
        scaler.scale(loss).backward()  # 缩放损失，避免梯度下溢
        # === [检查方法 3：在此处添加] ===修改（）
        scaler.unscale_(optimizer) # 在裁剪前必须 unscale
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        # === [检查结束] ===
        scaler.step(optimizer)  # 更新优化器（自动处理缩放）
        scaler.update()  # 更新缩放器状态

        # --------------------------
        # 记录损失
        # --------------------------
        batch_loss = loss.item()
        epoch_loss += batch_loss
        
        # 每50批或批次较少时打印日志
        if iteration % 50 == 0 or n_batches <= 10:
            print(
                f"==> Epoch[{epoch}]({iteration}/{n_batches}): "
                f"Loss: {batch_loss:.4f}", 
                flush=True
            )
            writer.add_scalar(
                'Train/BatchLoss', 
                batch_loss, 
                (epoch * n_batches) + iteration
            )

    # --------------------------
    # 步骤4：结束当前epoch
    # --------------------------
    avg_loss = epoch_loss / n_batches  # 计算epoch平均损失
    print(
        f"===> Epoch {epoch} 完成: 平均损失 {avg_loss:.4f}", 
        flush=True
    )
    writer.add_scalar('Train/AvgEpochLoss', avg_loss, epoch)
    
    return avg_loss  # 返回平均损失用于后续逻辑
    
# 为 infer_fusion_data 定义一个简单的 collate_fn
def collate_fn_inference(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return (None, None), None # 返回元组以匹配原逻辑

    (bev_range_data, indices) = zip(*batch)

    bev_imgs, range_imgs = zip(*bev_range_data)

    bev_imgs = torch.stack(bev_imgs, 0)
    range_imgs = torch.stack(range_imgs, 0)

    indices = list(indices)

    return (bev_imgs, range_imgs), indices

def infer_fusion_data(eval_set, model, opt, device):
    model.eval() 
    test_loader = DataLoader(dataset=eval_set, num_workers=opt.threads,
        batch_size=opt.cacheBatchSize, shuffle=False, 
        collate_fn=collate_fn_inference)
    all_global_descs = []
    with torch.no_grad():
        for data, indices in tqdm(test_loader, desc="提取BEV+Range融合特征"):
            if data is None: continue
            (imgs_bev, imgs_range) = data
            imgs_bev = imgs_bev.to(device)
            imgs_range = imgs_range.to(device)
            global_desc = model(imgs_bev, imgs_range)
            all_global_descs.append(global_desc.detach().cpu().numpy())
    if not all_global_descs:
        return np.array([]) # 返回空数组
    return np.concatenate(all_global_descs, axis=0)

def saveCheckpoint(state, is_best, model_out_path, filename='checkpoint.pth.tar'):
    if not exists(model_out_path): makedirs(model_out_path)
    filename = join(model_out_path, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, join(model_out_path, 'model_best.pth.tar'))
        print(f"===> 保存最佳模型到 {join(model_out_path, 'model_best.pth.tar')}")


if __name__ == "__main__":
    opt = get_args()

    print('====> FusionPlace (BEV+Range) (Dual Path) ====')
    import fusion_dataset as ds_module
    try:
        # <--- [V6.5 修复] ---
        # 导入新的、分离的字典
        from bevdata_dataset_for_radar import (
            bevdata_train_seq, bevdata_val_seq,bevdata_test_seq, 
            range_train_seq, range_val_seq, range_test_seq
        )
        # --- [修复结束] ---
    except ImportError:
        print("警告: 无法从 bevdata_dataset_for_radar.py 导入序列字典。")
        # <--- [V6.5 修复] ---
        # 确保 'except' 块也定义了 range_ 变量
        bevdata_train_seq = {"db": ["20231201_2"], "query": ["if_20240115_3"]} 
        bevdata_val_seq = {"db": ["iaf_20231201_2"], "query": ["iaf_20231201_3"]}
        range_train_seq = range_train_seq  # <-- [V6.5 修复] 解决 NameError
        range_val_seq = range_val_seq    # <-- [V6.5 修复] 解决 NameError
        # --- [修复结束] ---

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"===> 使用设备: {device}")
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    
    print('===> 加载 FusionPlaceModel')
    model = FusionPlaceModel(range_dim=opt.range_dim, freeze_backends=True)
    model = model.to(device)

    print('===> 加载 BEV 和 Range 预训练权重')
    if not isfile(opt.bev_weights_path):
         print(f"警告: BEV 权重 {opt.bev_weights_path} 不存在!")
    if not isfile(opt.diffloc_weights_path):
         print(f"警告: DiffLoc/DINO 权重 {opt.diffloc_weights_path} 不存在!")

    model.load_pretrained_weights(
        bev_weights_path=opt.bev_weights_path, 
        diffloc_weights_path=opt.diffloc_weights_path
    )
    print('===> 权重加载完成')

    # 训练模式
    if opt.mode.lower() == 'train':
        log_dir = join(opt.runsPath, f"fusion_v6_{datetime.now().strftime('%b%d_%H-%M-%S')}") # <-- (V6)
        writer = SummaryWriter(log_dir=log_dir)
        print(f"===> 训练日志保存路径: {log_dir}")

        with open(join(log_dir, 'train_config.json'), 'w') as f:
            json.dump(vars(opt), f, indent=2)

        # ==================== (V6.5 修复) ====================
        # (V6.5) 加载训练集
        bev_train_name = bevdata_train_seq["db"][0] 
        range_train_name = range_train_seq["db"][0]
        
        train_set = ds_module.TrainingDataset(
            bev_seq=bev_train_name,
            range_seq=range_train_name,
            bev_data_path=opt.bev_data_path,
            range_data_path=opt.range_data_path,
            range_data_type=opt.range_data_type,
            max_frames=3000,
            cache_path=opt.cachePath
        )
        # [V6.5 修复] 使用正确的变量名打印
        print(f"===> 训练集: BEV 序列 {bev_train_name} | Range 序列 {range_train_name} | 样本数 {len(train_set)}")

        # (V6.5) 加载验证集 Query
        bev_val_q_names = bevdata_val_seq["query"]
        range_val_q_names = range_val_seq["query"]
        # (假设 BEV 和 Range 验证集序列是一一对应的)
        assert len(bev_val_q_names) == len(range_val_q_names), "BEV 和 Range 验证集 Query 列表长度不匹配"
        
        val_sets_query = {bev_name: ds_module.InferDataset(
            bev_seq=bev_name,
            range_seq=range_name,
            bev_data_path=opt.bev_data_path,
            range_data_path=opt.range_data_path,
            range_data_type=opt.range_data_type,
            sample_inteval=opt.sample_interval
        ) for bev_name, range_name in zip(bev_val_q_names, range_val_q_names)}
        print(f"===> 验证集Query序列: {bev_val_q_names}, 共 {len(bev_val_q_names)} 个序列")

        # (V6.5) 加载验证集 DB
        bev_val_db_names = bevdata_val_seq["db"]
        range_val_db_names = range_val_seq["db"]
        assert len(bev_val_db_names) == len(range_val_db_names), "BEV 和 Range 验证集 DB 列表长度不匹配"

        val_sets_db = {bev_name: ds_module.InferDataset(
            bev_seq=bev_name,
            range_seq=range_name,
            bev_data_path=opt.bev_data_path,
            range_data_path=opt.range_data_path,
            range_data_type=opt.range_data_type,
            sample_inteval=opt.sample_interval
        ) for bev_name, range_name in zip(bev_val_db_names, range_val_db_names)}
        print(f"===> 验证集DB序列: {bev_val_db_names}, 共 {len(bev_val_db_names)} 个序列")
        # ================================================

        print("===> (注意) 优化器将只训练未冻结的层 (conv_stem, range_proj, fusion, bev_backbone.pooling)")
    # L518
        # === [关键修复 V3]：为 预训练层(pooling) 和 新层(stem, proj, fusion) 设置差分学习率 ===
        try:
            # 1. 识别 新的、从零开始训练的 层
            # [V3 修正]: 正确路径是 .rangevit.encoder.patch_embed
            new_params = list(model.range_feature_extractor.rangevit.encoder.patch_embed.parameters()) + \
                        list(model.range_proj.parameters()) + \
                        list(model.fusion.parameters())
            
            # 2. 识别 可训练的、但已预训练的 层
            pretrained_params = model.bev_backbone.pooling.parameters()
            
            # 3. 创建优化器参数组
            param_groups = [
                {'params': new_params, 'lr': opt.lr, 'weight_decay': opt.weightDecay}, 
                {'params': pretrained_params, 'lr': opt.lr * 0.1, 'weight_decay': opt.weightDecay} 
            ]
            
            print(f"===> (注意) 优化器将使用差分学习率:")
            print(f"     - 新层 (patch_embed, proj, fusion) LR: {opt.lr}")
            print(f"     - 预训练层 (pooling) LR: {opt.lr * 0.1}")
                
            optimizer = optim.Adam(param_groups)

        except Exception as e:
            print(f"警告: 设置差分学习率失败 ({e})。将回退到单一学习率。")
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=opt.lr,
                weight_decay=opt.weightDecay
            )
        # === [修复结束 V3] ===
        
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=opt.lrStep, gamma=opt.lrGamma
        )
        best_mean_recall = 0.0  

        # 训练轮次
        for epoch in range(opt.nEpochs):
            print(f"\n====> 训练轮次 {epoch+1}/{opt.nEpochs}")
            train_epoch(epoch, model, train_set, opt, device, writer, optimizer)

            # 验证阶段
            print(f"====> 验证轮次 {epoch+1}: 评估融合检索性能")
            val_recalls = []
            all_db_descs = [] 
            all_db_poses = [] 
            
            # [V6.5 修复] 迭代 val_sets_db.keys() (即 bev_val_db_names)
            for seq_name in val_sets_db.keys():
                db_descs = infer_fusion_data(val_sets_db[seq_name], model, opt, device).astype(np.float32)
                if db_descs.shape[0] == 0: continue # 跳过空序列
                db_pose = val_sets_db[seq_name].poses
                all_db_descs.append(db_descs)
                all_db_poses.append(db_pose)
                print(f"   - 已加载DB序列: {seq_name} | 特征维度: {db_descs.shape} | 位姿数量: {len(db_pose)}")

            if not all_db_descs:
                print("警告: 验证集DB中无有效数据，跳过此轮验证。")
                continue
                
            all_db_descs = np.concatenate(all_db_descs, axis=0)
            all_db_poses = np.concatenate(all_db_poses, axis=0)
            db_total_samples = len(all_db_poses)
            print(f"   => 合并后DB总特征: {all_db_descs.shape} | 总样本数: {db_total_samples}")
                    
            all_q_descs = [] 
            all_q_poses = [] 
            
            # [V6.5 修复] 迭代 val_sets_query.keys() (即 bev_val_q_names)
            for seq_name in val_sets_query.keys():
                q_descs = infer_fusion_data(val_sets_query[seq_name], model, opt, device).astype(np.float32)
                if q_descs.shape[0] == 0: continue # 跳过空序列
                q_pose = val_sets_query[seq_name].poses
                all_q_descs.append(q_descs)
                all_q_poses.append(q_pose)  
                print(f"   - 已加载Query序列: {seq_name} | 特征维度: {q_descs.shape} | 位姿数量: {len(q_pose)}")

            if not all_q_descs:
                print("警告: 验证集Query中无有效数据，跳过此轮验证。")
                continue

            all_q_descs = np.concatenate(all_q_descs, axis=0)
            all_q_poses = np.concatenate(all_q_poses, axis=0)
            q_total_samples = len(all_q_poses)
            print(f"   => 合并后Query总特征: {all_q_descs.shape} | 总样本数: {q_total_samples}")
                    
            class _SimpleEvalDataset:
                pass
            wrapper = _SimpleEvalDataset()
            wrapper.poses = np.concatenate([all_db_poses, all_q_poses], axis=0)
            wrapper.db_split_index = db_total_samples
            wrapper.sample_inteval = opt.sample_interval # <--- [V6.5 修复] 确保 evaluateResults 能访问

            # [V6.5 修复] 确保 ds_module.evaluateResults 被导入
            try:
                from bevdata_dataset_for_radar import evaluateResults
            except ImportError:
                print("警告: 无法导入 evaluateResults, Recall 将为 0.0")
                evaluateResults = lambda *args, **kwargs: (0.0, 0.0, 0.0, 0.0)

            recall_top1, _, _, _ = evaluateResults( 
                seq="all_db+all_query", 
                global_descs=[all_db_descs, all_q_descs],
                local_feats=None, 
                dataset=wrapper,
                match_results_save_path=None
            )
            val_recalls.append(recall_top1)
            writer.add_scalar(f'Val/Recall@Overall', recall_top1, epoch)
            print(f"===> 验证集总 Recall@1 {recall_top1:.4f}")

            mean_val_recall = np.mean(val_recalls) 
            writer.add_scalar('Val/MeanRecall', mean_val_recall, epoch)
            print(f"===> 验证集平均Recall@1: {mean_val_recall:.4f}")

            is_best = mean_val_recall > best_mean_recall
            if is_best:
                best_mean_recall = mean_val_recall
            
            save_state = {
                'epoch': epoch,
                'state_dict': model.state_dict(), 
                'mean_recall': mean_val_recall,
                'best_recall': best_mean_recall,
                'optimizer': optimizer.state_dict(),
            }
            saveCheckpoint(save_state, is_best, log_dir)

            lr_scheduler.step()
            print(f"===> 当前学习率: {optimizer.param_groups[0]['lr']:.6f}")

        print(f"\n====> 训练完成！最佳验证平均Recall@1: {best_mean_recall:.4f}")
        writer.close()

    # 测试模式
    elif opt.mode.lower() == 'test':
        print('===> 进入BEVData融合测试模式')
        if not exists(opt.match_save_path):
            makedirs(opt.match_save_path)
        print(f"===> 匹配结果保存路径: {opt.match_save_path}")

        test_seqs = [1] 
        test_recalls = []

        for seq in test_seqs:
            print(f"\n====> 测试序列: {seq}")
            
            # ==================== (V6.5 修复) ====================
            if opt.val_db_seq and opt.val_q_seq:
                db_set = ds_module.InferDataset(
                    bev_seq=opt.val_db_seq,       # <--- [修改]
                    range_seq=opt.val_db_seq,     # <--- [修改]
                    bev_data_path=opt.bev_data_path,
                    range_data_path=opt.range_data_path,
                    range_data_type=opt.range_data_type,
                    sample_inteval=opt.sample_interval
                )
                q_set = ds_module.InferDataset(
                    bev_seq=opt.val_q_seq,        # <--- [修改]
                    range_seq=opt.val_q_seq,      # <--- [修改]
                    bev_data_path=opt.bev_data_path,
                    range_data_path=opt.range_data_path,
                    range_data_type=opt.range_data_type,
                    sample_inteval=opt.sample_interval
                )
            # ================================================
                
                db_descs = infer_fusion_data(db_set, model, opt, device).astype(np.float32)
                q_descs = infer_fusion_data(q_set, model, opt, device).astype(np.float32)
                
                class _SimpleEvalDataset:
                    pass
                wrapper = _SimpleEvalDataset()
                wrapper.poses = np.concatenate([db_set.poses, q_set.poses], axis=0)
                wrapper.db_split_index = len(db_set.poses)
                wrapper.sample_inteval = opt.sample_interval # <--- [V6.5 修复]

                # [V6.5 修复] 确保 ds_module.evaluateResults 被导入
                try:
                    from bevdata_dataset_for_radar import evaluateResults
                except ImportError:
                    print("警告: 无法导入 evaluateResults, Recall 将为 0.0")
                    evaluateResults = lambda *args, **kwargs: (0.0, 0.0, 0.0, 0.0)

                recall_top1, _, _, _ = evaluateResults(
                    seq=f"{opt.val_db_seq}+{opt.val_q_seq}",
                    global_descs=[db_descs, q_descs],
                    local_feats=None, 
                    dataset=wrapper,
                    match_results_save_path=None 
                )
            else:
                print("警告: 融合模型测试不支持单序列模式。")
                print("请使用 --val_db_seq 和 --val_q_seq 指定DB和Query序列。")
                recall_top1 = 0.0 
            
            test_recalls.append(recall_top1)
            print(f"Recall@1: {recall_top1:.4f} ({recall_top1*100:.2f}%)")

        mean_recall = np.mean(test_recalls) if test_recalls else 0.0
        print(f"\n#################### BEVData测试总结果 ####################")
        print(f"平均Recall@1: {mean_recall:.4f} ({mean_recall*100:.2f}%)")
        print(f"###########################################################")

    else:
        raise ValueError(f"不支持的模式: {opt.mode}，仅支持 'train' 或 'test'")