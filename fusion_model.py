import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --- 从你的项目中导入骨干网络 ---
from REIN import REIN 
# <--- [修改 1/5] ---
# 我们现在严格遵循 DiffLoc 的导入
from image_feature_extractor import ImageFeatureExtractor
from model_utils import resize_pos_embed
# 移除 adapt_input_conv, 我们不再需要它
# --- [修改结束] ---

import torch.nn.functional as F

# --- (辅助函数 padding 保持不变) ---
def padding(im, patch_size):
    """
    (代码来自 diffloc/models/model_utils.py)
    """
    H, W = im.size(2), im.size(3)
    pH, pW = patch_size
    pad_h = (pH - H % pH) % pH
    pad_w = (pW - W % pW) % pW
    return F.pad(im, (0, pad_w, 0, pad_h))

# --- (CrossAttentionFusion 保持不变) ---
class CrossAttentionFusion(nn.Module):
    """
    交叉注意力融合模块 (BEV as Query, Range as Key/Value)
    """
    def __init__(self, d_model, nhead=4, num_layers=2):
        super().__init__()
        self.attention_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True) 
            for _ in range(num_layers)
        ])
        print(f"CrossAttentionFusion: d_model={d_model}, nhead={nhead}, num_layers={num_layers}")

    def forward(self, tgt, memory):
        output = tgt
        for layer in self.attention_layers:
            output = layer(output, memory)
        return output

# --- 核心的融合模型 (使用 NetVLAD) ---

class FusionPlaceModel(nn.Module):
    def __init__(self, range_dim=384, freeze_backends=True): # <--- [修改 2/5] range_dim 默认为 384 (ViT-S)
        """
        初始化融合模型 (NetVLAD 版本)。
        """
        super().__init__()
        
        # === 1. 加载 BEV 骨干网络 (来自 bevplace++) ===
        self.bev_backbone = REIN()
        self.bev_dim = 128 
        
        # === 2. 加载 Range 骨干网络 (来自 diffloc) ===
        # <--- [修改 3/5] ---
        # 严格遵循 DiffLoc 论文  和代码库的方案：
        # - conv_stem='v2' (或 'conv') 使用 models.stems.py 中的 ConvStem
        # - 这将替换掉 ViT 原生的 3 通道 patch_embed
        # - reuse_patch_emb=False 因为我们不再复用 DINO 的 patch_embed 权重
        print("✅ 正在初始化 'DiffLoc' 风格的 Range 特征提取器 (使用 ConvStem)...")
        self.range_feature_extractor = ImageFeatureExtractor(
            decoder='linear',
            conv_stem='v2',            # <-- 关键修改：使用 5 通道 ConvStem
            reuse_patch_emb=False,     # <-- 关键修改：不复用 DINO patch_embed
            in_channels=5              # <-- 明确指定输入5通道
        )
        # --- [修改结束] ---

        self.range_dim = range_dim # 384 (vit_small)

        # === 3. 冻结骨干网络 (关键步骤!) ===
        if freeze_backends:
            for param in self.bev_backbone.rem.parameters():
                param.requires_grad = False
                        
            # <--- [修改 4/5] ---
            # 我们需要冻结 ViT (rangevit.encoder)
            # 但 *不能* 冻结新加的 ConvStem (它在 encoder.patch_embed 里)
            print("正在冻结 Range-ViT Encoder (但保留 patch_embed 可训练)...")
            for name, param in self.range_feature_extractor.rangevit.encoder.named_parameters():
                # 检查参数名。如果它属于 'patch_embed' (即我们的 ConvStem)，
                # 我们就跳过它，让它保持可训练 (requires_grad=True)
                if 'patch_embed' in name:
                    print(f"  - (保持可训练) {name}")
                    continue
                
                # 否则，冻结所有其他层 (如 blocks, norm 等)
                param.requires_grad = False
            
            # (确保 DINO head 也不被训练, 如果它存在的话)
            # for param in self.range_feature_extractor.rangevit.head.parameters():
            #      param.requires_grad = False
                 
            print("✅ 骨干网络 (BEV-REM + Range-ViT Encoder) 已冻结。")
            print(" ---------------------------------------------------------")
            print(" ⚠️  以下层 *将会* 被训练 (这是正确的):")
            print("   1. self.range_feature_extractor.conv_stem (新的5通道输入层)")
            print("   2. self.range_proj (投影层)")
            print("   3. self.fusion (交叉注意力层)")
            print("   4. self.bev_backbone.pooling (NetVLAD 池化层)")
            print(" ---------------------------------------------------------")
            # --- [修改结束] ---
        
        # === 4. 定义新的融合模块 ===
        self.range_proj = nn.Linear(self.range_dim, self.bev_dim)
        self.fusion = CrossAttentionFusion(d_model=self.bev_dim, nhead=4, num_layers=2)
        
        # === 5. 我们将使用 bev_backbone.pooling ===
        self.global_feat_dim = self.bev_dim * 64 
        
        print(f"✅ FusionPlaceModel (with NetVLAD) 初始化成功!")
        print(f"   - BEV 特征维度 (Q): {self.bev_dim}")
        print(f"   - Range 特征维度 (K,V): {self.range_dim} -> 投影到 {self.bev_dim}")
        print(f"   - 池化层: REIN 原生 NetVLAD")
        print(f"   - 最终描述符维度: {self.global_feat_dim}")

    def load_pretrained_weights(self, bev_weights_path, diffloc_weights_path):
        """
        加载两个模型的预训练权重。
        [V6 DiffLoc 改进版]
        """
        
        # 1. 加载 BEVPlace++ (REIN) 权重
        print(f"正在加载 BEV 权重: {bev_weights_path}")
        try:
            bev_ckpt = torch.load(bev_weights_path, map_location='cpu', weights_only=False)
            if 'state_dict' in bev_ckpt:
                self.bev_backbone.load_state_dict(bev_ckpt['state_dict'])
                print("BEV 权重 (包含 NetVLAD) 加载完毕。")
            else:
                print(f"错误: BEV 权重文件 {bev_weights_path} 中未找到 'state_dict'。")
        except Exception as e:
            print(f"加载 BEV 权重失败: {e}。将跳过。")

        # 2. 加载 Range 权重 (DINO)
        print(f"正在加载 Range 权重 (DINO): {diffloc_weights_path}")
        try:
            diffloc_ckpt = torch.load(diffloc_weights_path, map_location='cpu', weights_only=False)

            dict_to_search = {}
            if 'model_state' in diffloc_ckpt: dict_to_search = diffloc_ckpt['model_state']
            elif 'model' in diffloc_ckpt: dict_to_search = diffloc_ckpt['model']
            else: dict_to_search = diffloc_ckpt

            # <--- [修改 5/5] 
            # 这是“严格遵循” DiffLoc 仓库的代码逻辑
            
            # --- [核心：剥离前缀] ---
            # (这个逻辑不变，假设你的DINO权重文件也是这样)
            prefix_to_strip = 'image_feature_extractor.rangevit.'
            range_encoder_weights = {}
            
            for k, v in dict_to_search.items():
                if 'head' in k: # 跳过 DINO 分类头
                    continue
                
                if k.startswith(prefix_to_strip):
                    new_k = k[len(prefix_to_strip):]
                    range_encoder_weights[new_k] = v
            
            if not range_encoder_weights:
                 print(f"警告: 无法从 {diffloc_weights_path} 中加载任何DINO权重 (前缀剥离后为空)。")
                 # (尝试不剥离前缀)
                 for k, v in dict_to_search.items():
                    if 'head' in k: continue
                    range_encoder_weights[k] = v
                 print("...已尝试不剥离前缀，重新加载。")


            # --- [核心：替换方案] ---
            # 1. 移除 DINO 的 patch_embed 权重, 因为我们不用它
            #    (我们用的是新的、从头训练的 ConvStem)
            keys_to_delete = ['patch_embed.proj.weight', 'patch_embed.proj.bias']
            for k in keys_to_delete:
                if k in range_encoder_weights:
                    del range_encoder_weights[k]
                    print(f"已从DINO权重中移除 '{k}' (将被 ConvStem 替换)")

            # 2. 适配 Positional Embedding (如果需要)
            # (这部分逻辑与 V5 相同，因为 pos_embed 仍在 ViT 内部)
            if 'encoder.pos_embed' in range_encoder_weights:
                pos_embed_ckpt = range_encoder_weights['encoder.pos_embed']
                
                # (假设我们的 ConvStem 输出的 grid_size 也是 (8, 32))
                # (这取决于 stems.py 中 ConvStem 的下采样率)
                # (DiffLoc 默认的 (32, 512)@P(4,16) -> (8, 32) grid)
                gs_new_h, gs_new_w = 8, 32 
                
                target_shape_dim1 = gs_new_h * gs_new_w + 1
                if pos_embed_ckpt.shape[1] != target_shape_dim1:
                    print(f"正在将 pos_embed 从 {pos_embed_ckpt.shape} 重采样到 (1, {target_shape_dim1}, {self.range_dim})...")
                    resized_pos_emb = resize_pos_embed(
                        pos_embed_ckpt,
                        grid_old_shape=None, 
                        grid_new_shape=(gs_new_h, gs_new_w),
                        num_extra_tokens=1
                    )
                    range_encoder_weights['encoder.pos_embed'] = resized_pos_emb
                else:
                     print(f"pos_embed 形状 {pos_embed_ckpt.shape} 已匹配目标，跳过重采样。")
            
            # 3. (已移除) 不再需要 F.interpolate 和 adapt_input_conv

            # --- [核心：结束] ---

            # 加载权重, strict=False 因为 patch_embed 丢失了
            msg = self.range_feature_extractor.rangevit.load_state_dict(range_encoder_weights, strict=False)
            
            print(f"Range 权重 (DINO ViT blocks) 加载完毕:")
            print(f"  - 丢失的键 (Missing keys): {msg.missing_keys}")
            print(f"  - 意外的键 (Unexpected keys): {msg.unexpected_keys}")

        except Exception as e:
            print(f"加载 Range 权重失败: {e}。将跳过。")

    # --- (forward 函数保持不变) ---
    def forward(self, bev_input, range_input):
        """
        模型的前向传播。
        """
        
        # 冻结的骨干网络应处于评估模式 (关闭dropout等)
        if hasattr(self, 'freeze_backends') and self.freeze_backends:
            self.bev_backbone.rem.eval()
            self.range_feature_extractor.rangevit.encoder.eval() # <--- 明确冻结 encoder
            #self.range_feature_extractor.rangevit.head.eval()   # <--- 明确冻结 head
            # (ConvStem 保持 .train() 模式，如果它有 BN 层)

        # === 1. 提取 BEV 特征 (Query) ===
        bev_feature_map, _ = self.bev_backbone.rem(bev_input)

        # === 2. 提取 Range 特征 (Key, Value) ===
        # (DiffLoc 的 ImageFeatureExtractor 内部会处理 padding 和 ConvStem)
        
        # x shape: [B, num_tokens + 1 (CLS), 384]
        x = self.range_feature_extractor(range_input) 
        
        # range_tokens shape: [B, num_tokens, 384]
        range_tokens = x[:, 1:, :] # 去掉 [CLS] token
        
        # === 3. 交叉注意力融合 ===
        B, C_bev, H_bev_feat, W_bev_feat = bev_feature_map.shape
        bev_q = bev_feature_map.flatten(2).permute(0, 2, 1) # [B, H*W, 128]
        
        range_kv = self.range_proj(range_tokens) # [B, N_tokens, 128]
        
        fused_feature_seq = self.fusion(tgt=bev_q, memory=range_kv)
        
        fused_feature_map = fused_feature_seq.permute(0, 2, 1).view(B, C_bev, H_bev_feat, W_bev_feat)
        
        # === 4. 添加残差连接 ===
        enhanced_bev_map = bev_feature_map + fused_feature_map
        
        # === 5. (已修改) 使用 REIN 原生的 NetVLAD 池化 ===
        final_descriptor = self.bev_backbone.pooling(enhanced_bev_map)
        
        return final_descriptor