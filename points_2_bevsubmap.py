import os
import csv


from typing import Dict, List, Optional, Tuple

import concurrent.futures as cf

import numpy as np
import open3d as o3d


# 在 numpy 导入后尝试对 OpenBLAS 做额外限制（如果可用）
np_config = np.__config__.show()
if "openblas" in str(np_config).lower():
    # 如果使用 OpenBLAS，设置其线程数
    import ctypes

    try:
        openblas = ctypes.CDLL("libopenblas.so")
        openblas.openblas_set_num_threads(1)
    except Exception:
        pass


try:
    from tqdm import tqdm

    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False


import pair_radar_lidar_nearest_frame as base


def _rotation_matrix_from_rpy_xyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """根据 SciPy 的 'xyz' 欧拉角约定重建旋转矩阵：R = Rx(roll) @ Ry(pitch) @ Rz(yaw)。"""

    sr, cr = np.sin(roll), np.cos(roll)
    sp, cp = np.sin(pitch), np.cos(pitch)
    sy, cy = np.sin(yaw), np.cos(yaw)

    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=float)
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=float)
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    return (Rx @ Ry) @ Rz


def load_gps_pose_entries(csv_path: str) -> List[Dict]:
    """从 gps.csv 中加载位姿，生成与 base.load_transformation_entries 相同结构的数据。"""

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"GPS pose file not found: {csv_path}")

    entries: List[Dict] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required_fields = {"timestamp", "northing", "easting", "height", "roll", "pitch", "yaw"}
        if reader.fieldnames is None or not required_fields.issubset(set(reader.fieldnames)):
            raise ValueError(
                "GPS CSV 缺少必要字段，期望包含: timestamp,northing,easting,height,roll,pitch,yaw"
            )

        for row in reader:
            try:
                ts = float(row["timestamp"])
                northing = float(row["northing"])
                easting = float(row["easting"])
                height = float(row["height"])
                roll = float(row["roll"])
                pitch = float(row["pitch"])
                yaw = float(row["yaw"])
            except (TypeError, ValueError):
                continue

            T = np.eye(4, dtype=float)
            T[:3, :3] = _rotation_matrix_from_rpy_xyz(roll, pitch, yaw)
            # 假设 east->x, north->y, height->z，与原 UTM 位姿文件保持一致
            T[:3, 3] = np.array([easting, northing, height], dtype=float)

            entries.append(
                {
                    "timestamp": ts,
                    "T": T,
                    "t": T[:3, 3].copy(),
                    "rpy": (roll, pitch, yaw),
                }
            )

    entries.sort(key=lambda item: item["timestamp"])
    return entries


def compute_bev_grid(
    points: np.ndarray,
    x_range_m: Tuple[float, float],
    y_range_m: Tuple[float, float],
    resolution_m: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """计算点云在 BEV 网格中的行列坐标与 ROI 掩码。"""

    xmin, xmax = float(x_range_m[0]), float(x_range_m[1])
    ymin, ymax = float(y_range_m[0]), float(y_range_m[1])
    res = float(max(resolution_m, 1e-9))

    H = int(np.ceil((xmax - xmin) / res))
    W = int(np.ceil((ymax - ymin) / res))

    if points.size == 0:
        empty_idx = np.zeros((0,), dtype=np.int32)
        empty_mask = np.zeros((0,), dtype=bool)
        return empty_idx, empty_idx, empty_mask, H, W

    x = points[:, 0]
    y = points[:, 1]

    mask = (x >= xmin) & (x < xmax) & (y >= ymin) & (y < ymax)
    if not np.any(mask):
        empty_idx = np.zeros((0,), dtype=np.int32)
        return empty_idx, empty_idx, mask, H, W

    x_roi = x[mask]
    y_roi = y[mask]

    ix = np.floor((x_roi - xmin) / res).astype(np.int32)
    iy = np.floor((y_roi - ymin) / res).astype(np.int32)

    rows = (H - 1 - ix).clip(0, H - 1)
    cols = iy.clip(0, W - 1)
    return rows, cols, mask, H, W


def aggregate_height_map(
    points_roi: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    height_px: int,
    width_px: int,
    fill_value: float = np.nan,
) -> np.ndarray:
    """根据 ROI 内点集生成最大高度图。"""

    if points_roi.size > 0 and points_roi.shape[1] < 3:
        raise ValueError("points_roi 至少需要包含 z 维度")

    height_map = np.full((height_px, width_px), -np.inf, dtype=np.float32)
    if rows.size > 0:
        z_vals = points_roi[:, 2].astype(np.float32, copy=False)
        flat = height_map.ravel()
        lin_idx = rows.astype(np.int64) * width_px + cols.astype(np.int64)
        np.maximum.at(flat, lin_idx, z_vals)
        height_map = flat.reshape(height_px, width_px)

    if np.isnan(fill_value):
        height_map[~np.isfinite(height_map)] = np.nan
    else:
        height_map[~np.isfinite(height_map)] = float(fill_value)
    return height_map


def aggregate_count_map(
    rows: np.ndarray,
    cols: np.ndarray,
    height_px: int,
    width_px: int,
) -> np.ndarray:
    """生成每个网格内点数量的计数图。"""

    count_map = np.zeros((height_px, width_px), dtype=np.uint32)
    if rows.size > 0:
        flat = count_map.ravel()
        lin_idx = rows.astype(np.int64) * width_px + cols.astype(np.int64)
        np.add.at(flat, lin_idx, 1)
        count_map = flat.reshape(height_px, width_px)
    return count_map


def _compute_integral_image(mask_2d: np.ndarray) -> np.ndarray:
    """计算二值掩码的积分图（1-based padding）。"""

    if mask_2d.ndim != 2:
        raise ValueError("mask_2d 必须是二维数组")
    H, W = mask_2d.shape
    ii = np.zeros((H + 1, W + 1), dtype=np.int32)
    if H > 0 and W > 0:
        ii[1:, 1:] = np.cumsum(np.cumsum(mask_2d.astype(np.int32), axis=0), axis=1)
    return ii


def compute_radar_acceptance(
    rows: np.ndarray,
    cols: np.ndarray,
    lidar_mask: np.ndarray,
    resolution_m: float,
    max_dist_m: float,
) -> np.ndarray:
    """根据 LiDAR 占据掩码筛选雷达点的有效性。"""

    if rows.size == 0:
        return np.zeros((0,), dtype=bool)

    if lidar_mask.size == 0 or np.count_nonzero(lidar_mask) == 0:
        # 若 LiDAR 掩码为空，直接全部保留
        return np.ones(rows.shape[0], dtype=bool)

    res = float(max(resolution_m, 1e-9))
    r_px = int(np.ceil(float(max_dist_m) / res))

    if r_px <= 0:
        return lidar_mask[rows, cols] > 0

    H, W = lidar_mask.shape
    ii = _compute_integral_image(lidar_mask)

    top = np.maximum(rows - r_px, 0)
    left = np.maximum(cols - r_px, 0)
    bottom = np.minimum(rows + r_px, H - 1)
    right = np.minimum(cols + r_px, W - 1)

    sums = ii[bottom + 1, right + 1] - ii[top, right + 1] - ii[bottom + 1, left] + ii[top, left]
    return sums > 0


def run_for_base_dir(base_dir: str) -> None:
    preprocessed_base_dir = f"{base_dir}_preprocessed"
    lidar_dir = f"{preprocessed_base_dir}/lidarpoints"
    radar_dir = f"{preprocessed_base_dir}/pointclouds"
    gps_csv_path = f"{preprocessed_base_dir}/gps.csv"
    lidar_calib_path = f"{base_dir}/body_T_xt32.txt"
    radar_calib_path = f"{base_dir}/body_T_oculii.txt"

    if not os.path.isdir(preprocessed_base_dir):
        raise RuntimeError(f"预处理目录不存在: {preprocessed_base_dir}")
    if not os.path.isdir(lidar_dir):
        raise RuntimeError(f"LiDAR 目录不存在: {lidar_dir}")
    if not os.path.isdir(radar_dir):
        raise RuntimeError(f"Radar 目录不存在: {radar_dir}")

    # --- 参数配置 ---
    submap_window_size = 7         # 雷达子地图聚合窗口帧数，需为奇数
    lidar_submap_window_size = 7   # LiDAR 子地图聚合窗口帧数，需为奇数
    submap_stride = 1              # 相邻 submap 之间的中心帧步长

    max_range_m = 120.0
    hfov_deg = 110.0
    apply_lidar_fov_crop = True
    apply_lidar_near_radar_filter = True
    lidar_distance_filter_mode = False
    lidar_near_radar_max_dist_m = 15.0
    apply_radar_near_lidar_supervision = False
    radar_near_lidar_max_dist_m = 8.0

    max_time_diff = 0.1

    bev_resolution_m = 0.5
    bev_x_range_m = (0.0, 120.0)
    bev_y_range_m = (-60.0, 60.0)

    generate_lidar_zmax = True
    generate_lidar_count = False
    generate_radar_zmax = True
    generate_radar_count = False

    if lidar_distance_filter_mode:
        apply_lidar_near_radar_filter = True
        generate_radar_zmax = False
        generate_radar_count = False

    batch_num_workers = 4

    if submap_window_size <= 0 or submap_window_size % 2 == 0:
        raise ValueError("submap_window_size 必须为正奇数")
    if lidar_submap_window_size <= 0 or lidar_submap_window_size % 2 == 0:
        raise ValueError("lidar_submap_window_size 必须为正奇数")

    submap_half = submap_window_size // 2
    lidar_submap_half = lidar_submap_window_size // 2

    print("Loading GPS-based poses...")
    entries = load_gps_pose_entries(gps_csv_path)
    if not entries:
        raise RuntimeError("位姿文件为空或无法解析有效条目。")

    print("Loading extrinsics and computing L_T_R (LiDAR->Radar)...")
    T_body_lidar = np.loadtxt(lidar_calib_path, dtype=float).reshape(4, 4)
    T_body_radar = np.loadtxt(radar_calib_path, dtype=float).reshape(4, 4)
    L_T_R = np.linalg.inv(T_body_lidar) @ T_body_radar

    for entry in entries:
        U_T_body = entry["T"]
        entry["T_body"] = U_T_body
        entry["T_lidar"] = U_T_body @ T_body_lidar
        entry["T_radar"] = U_T_body @ T_body_radar

    radar_files_all = sorted(
        [f for f in os.listdir(radar_dir) if f.endswith(".npy")],
        key=lambda f: float(os.path.splitext(f)[0]),
    )
    if not radar_files_all:
        raise RuntimeError("Radar 目录下未找到 .npy 文件。")

    if len(radar_files_all) < submap_window_size:
        print(
            f"可用雷达帧数量不足以构建窗口 (需要 {submap_window_size}, 实际 {len(radar_files_all)})，跳过。"
        )
        return

    lidar_files_all = sorted(
        [f for f in os.listdir(lidar_dir) if f.endswith(".pcd") or f.endswith(".npy")],
        key=lambda f: float(os.path.splitext(f)[0]),
    )
    if not lidar_files_all:
        raise RuntimeError("LiDAR 目录下未找到 .pcd/.npy 文件。")

    if len(lidar_files_all) < lidar_submap_window_size:
        print(
            f"可用 LiDAR 帧数量不足以构建窗口 (需要 {lidar_submap_window_size}, 实际 {len(lidar_files_all)})，跳过。"
        )
        return

    lidar_times_all = np.array([float(os.path.splitext(f)[0]) for f in lidar_files_all])

    lidar_submap_zmax_dir = os.path.join(preprocessed_base_dir, "lidar_submap_zmax")
    lidar_submap_count_dir = os.path.join(preprocessed_base_dir, "lidar_submap_count")
    radar_submap_zmax_dir = os.path.join(preprocessed_base_dir, "radar_submap_zmax")
    radar_submap_count_dir = os.path.join(preprocessed_base_dir, "radar_submap_count")

    if generate_lidar_zmax:
        os.makedirs(lidar_submap_zmax_dir, exist_ok=True)
    if generate_lidar_count:
        os.makedirs(lidar_submap_count_dir, exist_ok=True)
    if generate_radar_zmax:
        os.makedirs(radar_submap_zmax_dir, exist_ok=True)
    if generate_radar_count:
        os.makedirs(radar_submap_count_dir, exist_ok=True)

    submap_indices = list(range(submap_half, len(radar_files_all) - submap_half, submap_stride))
    if not submap_indices:
        print("未生成任何 submap 中心索引，检查窗口与步长配置。")
        return

    def process_one_submap(center_idx: int) -> int:
        try:
            center_file = radar_files_all[center_idx]
            ts_center = float(os.path.splitext(center_file)[0])
        except Exception:
            return 0

        center_pose = base.find_nearest_entry(ts_center, entries)
        if center_pose is None:
            return 0
        if (max_time_diff is not None) and (
            abs(center_pose["timestamp"] - ts_center) > max_time_diff
        ):
            return 0

        U_T_L_center = center_pose.get("T_lidar")
        if U_T_L_center is None:
            U_T_body_center = center_pose["T"]
            U_T_L_center = U_T_body_center @ T_body_lidar
            center_pose["T_lidar"] = U_T_L_center
        U_T_R_center = center_pose.get("T_radar")
        if U_T_R_center is None:
            U_T_body_center = center_pose["T"]
            U_T_R_center = U_T_body_center @ T_body_radar
            center_pose["T_radar"] = U_T_R_center

        window_start = center_idx - submap_half
        window_end = center_idx + submap_half

        radar_pts_world_list: List[np.ndarray] = []
        for ridx in range(window_start, window_end + 1):
            rf = radar_files_all[ridx]
            rp = os.path.join(radar_dir, rf)
            try:
                ts_r = float(os.path.splitext(rf)[0])
            except Exception:
                continue

            pose_r = base.find_nearest_entry(ts_r, entries)
            if pose_r is None:
                continue
            if (max_time_diff is not None) and (
                abs(pose_r["timestamp"] - ts_r) > max_time_diff
            ):
                continue

            pts_radar, _, _ = base.load_radar_points_and_attrs(rp)
            if pts_radar.size == 0:
                continue

            U_T_R_i = pose_r.get("T_radar")
            if U_T_R_i is None:
                U_T_body_r = pose_r["T"]
                U_T_R_i = U_T_body_r @ T_body_radar
                pose_r["T_radar"] = U_T_R_i
            pts_world = base.transform_points(U_T_R_i, pts_radar)
            radar_pts_world_list.append(pts_world)

        if radar_pts_world_list:
            radar_pts_world = np.concatenate(radar_pts_world_list, axis=0)
            mask_radar_fov = base.mask_points_in_radar_fov(
                radar_pts_world,
                U_T_R_center,
                max_range_m=max_range_m,
                hfov_deg=hfov_deg,
            )
            radar_pts_world = radar_pts_world[np.where(mask_radar_fov)[0]]
        else:
            radar_pts_world = np.empty((0, 3), dtype=float)

        if lidar_times_all.size == 0:
            return 0

        center_l_idx = int(np.argmin(np.abs(lidar_times_all - ts_center)))
        lidar_acc_world = o3d.geometry.PointCloud()

        start_l = max(0, center_l_idx - lidar_submap_half)
        end_l = min(len(lidar_files_all) - 1, center_l_idx + lidar_submap_half)

        for j in range(start_l, end_l + 1):
            lf = lidar_files_all[j]
            lp = os.path.join(lidar_dir, lf)
            try:
                ts_l = float(os.path.splitext(lf)[0])
            except Exception:
                continue

            pose_l = base.find_nearest_entry(ts_l, entries)
            if pose_l is None:
                continue
            if (max_time_diff is not None) and (
                abs(pose_l["timestamp"] - ts_l) > max_time_diff
            ):
                continue

            U_T_L = pose_l.get("T_lidar")
            if U_T_L is None:
                U_T_body_l = pose_l["T"]
                U_T_L = U_T_body_l @ T_body_lidar
                pose_l["T_lidar"] = U_T_L
            if lp.lower().endswith(".npy"):
                arr = np.load(lp)
                if arr.ndim != 2 or arr.shape[1] < 3:
                    continue
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(arr[:, :3].astype(float, copy=False))
            else:
                pcd = base.load_point_cloud(lp)

            pcd_world = o3d.geometry.PointCloud(pcd)
            pcd_world.transform(U_T_L)
            lidar_acc_world += pcd_world

        if apply_lidar_fov_crop and lidar_acc_world.has_points():
            pts_lidar_world = np.asarray(lidar_acc_world.points)
            mask_lidar = base.mask_points_in_radar_fov(
                pts_lidar_world,
                U_T_R_center,
                max_range_m=max_range_m,
                hfov_deg=hfov_deg,
            )
            lidar_acc_world = lidar_acc_world.select_by_index(np.where(mask_lidar)[0])

        lidar_pts_world = (
            np.asarray(lidar_acc_world.points)
            if lidar_acc_world.has_points()
            else np.empty((0, 3))
        )

        T_radar_world_center = np.linalg.inv(U_T_R_center)
        lidar_pts_radar = (
            base.transform_points(T_radar_world_center, lidar_pts_world)
            if lidar_pts_world.size > 0
            else np.empty((0, 3))
        )
        radar_pts_radar = (
            base.transform_points(T_radar_world_center, radar_pts_world)
            if radar_pts_world.size > 0
            else np.empty((0, 3))
        )

        if (
            apply_lidar_near_radar_filter
            and lidar_pts_radar.size > 0
            and radar_pts_radar.size > 0
        ):
            lidar_pts_2d = lidar_pts_radar.copy()
            radar_pts_2d = radar_pts_radar.copy()
            lidar_pts_2d[:, 2] = 0.0
            radar_pts_2d[:, 2] = 0.0

            lidar_pcd = o3d.geometry.PointCloud()
            lidar_pcd.points = o3d.utility.Vector3dVector(lidar_pts_2d)
            radar_pcd = o3d.geometry.PointCloud()
            radar_pcd.points = o3d.utility.Vector3dVector(radar_pts_2d)

            dists = lidar_pcd.compute_point_cloud_distance(radar_pcd)
            dists = np.asarray(dists, dtype=float)
            idx_keep = np.where(dists <= lidar_near_radar_max_dist_m)[0]
            lidar_pts_radar = lidar_pts_radar[idx_keep]

        if (
            apply_lidar_near_radar_filter
            and radar_pts_radar.size == 0
        ):
            lidar_pts_radar = np.empty((0, 3))

        lidar_rows, lidar_cols, lidar_mask_roi, H, W = compute_bev_grid(
            lidar_pts_radar, bev_x_range_m, bev_y_range_m, bev_resolution_m
        )
        lidar_pts_roi = lidar_pts_radar[lidar_mask_roi]
        lidar_occ_mask = np.zeros((H, W), dtype=np.uint8)
        if lidar_rows.size > 0:
            lidar_occ_mask[lidar_rows, lidar_cols] = 1

        radar_rows, radar_cols, radar_mask_roi, H_r, W_r = compute_bev_grid(
            radar_pts_radar, bev_x_range_m, bev_y_range_m, bev_resolution_m
        )
        if H_r != H or W_r != W:
            raise ValueError("雷达与 LiDAR BEV 网格尺寸不一致")
        radar_pts_roi = radar_pts_radar[radar_mask_roi]

        if apply_radar_near_lidar_supervision and radar_rows.size > 0:
            acceptance = compute_radar_acceptance(
                radar_rows,
                radar_cols,
                lidar_occ_mask,
                bev_resolution_m,
                radar_near_lidar_max_dist_m,
            )
            if np.any(acceptance):
                radar_pts_roi = radar_pts_roi[acceptance]
                radar_rows = radar_rows[acceptance]
                radar_cols = radar_cols[acceptance]
            else:
                radar_pts_roi = radar_pts_roi[:0]
                radar_rows = radar_rows[:0]
                radar_cols = radar_cols[:0]

        out_name = f"{ts_center:.9f}_res{bev_resolution_m:.2f}m.npy"

        if generate_lidar_zmax:
            lidar_bev_zmax = aggregate_height_map(
                lidar_pts_roi, lidar_rows, lidar_cols, H, W, fill_value=np.nan
            )
            np.save(os.path.join(lidar_submap_zmax_dir, out_name), lidar_bev_zmax)

        if generate_lidar_count:
            lidar_bev_count = aggregate_count_map(lidar_rows, lidar_cols, H, W)
            Nm = int(lidar_bev_count.max()) if lidar_bev_count.size > 0 else 0
            if Nm > 0:
                lidar_bev_intensity = (np.minimum(lidar_bev_count, Nm) / float(Nm)).astype(np.float32)
            else:
                lidar_bev_intensity = np.zeros_like(lidar_bev_count, dtype=np.float32)
            np.save(os.path.join(lidar_submap_count_dir, out_name), lidar_bev_intensity)

        if generate_radar_zmax:
            radar_bev_zmax = aggregate_height_map(
                radar_pts_roi, radar_rows, radar_cols, H, W, fill_value=np.nan
            )
            np.save(os.path.join(radar_submap_zmax_dir, out_name), radar_bev_zmax)

        if generate_radar_count:
            radar_bev_count = aggregate_count_map(radar_rows, radar_cols, H, W)
            Nm_r = int(radar_bev_count.max()) if radar_bev_count.size > 0 else 0
            if Nm_r > 0:
                radar_bev_intensity = (np.minimum(radar_bev_count, Nm_r) / float(Nm_r)).astype(np.float32)
            else:
                radar_bev_intensity = np.zeros_like(radar_bev_count, dtype=np.float32)
            np.save(os.path.join(radar_submap_count_dir, out_name), radar_bev_intensity)

        return 1

    saved_submaps = 0
    iterator = submap_indices
    pbar = None
    if _HAS_TQDM:
        pbar = tqdm(total=len(iterator), desc="BEV submap", unit="submap")

    with cf.ThreadPoolExecutor(max_workers=max(1, int(batch_num_workers))) as executor:
        futures = [executor.submit(process_one_submap, idx) for idx in iterator]
        for fut in cf.as_completed(futures):
            try:
                saved_submaps += int(fut.result() or 0)
            except Exception:
                pass
            if pbar is not None:
                pbar.update(1)

    if pbar is not None:
        pbar.close()

    print(
        "Submap generation finished. Saved {cnt} submaps.".format(cnt=saved_submaps)
    )


def main():
    base_root_dir = "/root/autodl-tmp/snail-radar"
    dataset_dict: Dict[str, List[str]] = {
    # "bc": [
    #     "20230920/1",
    #     "20230921/2",
    #     "20231007/4",
    #     "20231105/6",
    #     "20231105_aft/2"
    # ],
    # "sl": [
    #     # "20230920/2",
    #     # "20230921/3",
    #     # "20230921/5",
    #     # "20231007/2",
    #     "20231019/1",
    #     "20231105/2",
    #     "20231105/3",
    #     "20231105_aft/4",
    #     "20231109/3"
    # ],
    # "ss": [
    #     "20230921/4",
    #     "20231019/2",
    #     # "20231105/4",
    #     # "20231105/5",
    #     "20231105_aft/5",
    #     "20231109/4"
    # ],
    # "if": [
    #     # "20231208/4",
    #     "20231213/4",
    #     # "20231213/5",
    #     "20240115/3",
    #     # "20240116/5",
    #     # "20240116_eve/5",
    #     # "20240123/3"
    # ],
    # "iaf": [
    #     "20231201/2",
    #     "20231201/3",
    #     # "20231208/5",
    #     # "20231213/2",
    #     # "20231213/3",
    #     # "20240113/2",
    #     # "20240113/3",
    #     # "20240116_eve/4"
    # ],
    # "iaef": [
    #     "20240113/5",
    #     "20240115/2",
    #     "20240116/4"
    # ],
    # # "st": [
    # #     "20231208/1",
    # #     "20231213/1",
    # #     "20240113/1"
    # # ],
    "81r": [
        # "20240116/2",
        "20240116_eve/3",
        # "20240123/2"
    ]
}

    if len(dataset_dict) == 0:
        base_dirs = ["/root/autodl-tmp/snail-radar/81r/20240116_2"]
    else:
        base_dirs = []
        for key, subsets in dataset_dict.items():
            for subset in subsets:
                subset_dir = subset.replace("/", "_")
                base_dirs.append(f"{base_root_dir}/{key}/{subset_dir}")

    for base_dir in base_dirs:
        print(f"\n==== Processing dataset: {base_dir} ====")
        try:
            run_for_base_dir(base_dir)
        except Exception as exc:
            print(f"[WARN] Skip dataset {base_dir} due to error: {exc}")


if __name__ == "__main__":
    main()


