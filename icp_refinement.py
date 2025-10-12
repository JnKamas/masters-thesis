#!/usr/bin/env python3
import os
import argparse
import numpy as np
import open3d as o3d
import cv2
from tqdm import tqdm


def read_transform_file(file):
    """Read 4x4 transform (prediction) file; accepts commas or spaces."""
    with open(file, "r") as f:
        line = f.readline().strip().replace(",", " ")
        P = line.split()
    R = np.array([[float(P[0]), float(P[4]), float(P[8])],
                  [float(P[1]), float(P[5]), float(P[9])],
                  [float(P[2]), float(P[6]), float(P[10])]])
    t = np.array([float(P[12]), float(P[13]), float(P[14])])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def write_transform_file(file, T):
    """Write matrix in CSV style compatible with evaluator."""
    with open(file, "w") as f:
        line = (
            f"{T[0,0]}, {T[1,0]}, {T[2,0]}, 0.0, "
            f"{T[0,1]}, {T[1,1]}, {T[2,1]}, 0.0, "
            f"{T[0,2]}, {T[1,2]}, {T[2,2]}, 0.0, "
            f"{T[0,3]}, {T[1,3]}, {T[2,3]}, 1.0"
        )
        f.write(line)


def load_pointcloud_from_exr(exr_path, voxel_size=0.5):
    """Load point cloud from EXR scene depth file."""
    xyz = cv2.imread(exr_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if xyz is None:
        raise ValueError(f"Could not load {exr_path}")
    xyz = xyz.reshape(-1, 3)
    xyz = xyz[~np.isnan(xyz).any(axis=1)]
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=2.0, max_nn=30)
    )
    return pcd


def load_model_from_stl(stl_path, voxel_size=1.0, n_points=100000):
    """Load model mesh and sample it into a point cloud."""
    mesh = o3d.io.read_triangle_mesh(stl_path)
    if mesh.is_empty():
        raise ValueError(f"Could not load STL: {stl_path}")
    pcd = mesh.sample_points_uniformly(number_of_points=n_points)
    # auto-scale if STL is in meters
    extent = np.max(pcd.get_axis_aligned_bounding_box().get_extent())
    if extent < 10:  # likely in meters
        pcd.scale(1000.0, center=(0, 0, 0))
    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=2.0, max_nn=30)
    )
    return pcd


def run_icp(source, target, threshold, init_transform, max_iter=80):
    """Run 2-stage ICP (coarse point-to-point, fine point-to-plane) and return relative correction."""
    if max_iter == 0:
        return np.eye(4), None

    # Stage 1: coarse alignment – large threshold, point-to-point
    result = o3d.pipelines.registration.registration_icp(
        source, target, threshold, init_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter // 2)
    )

    # Compute relative correction (ΔT = T_refined * inv(T_init))
    T_refined = result.transformation
    T_rel = T_refined @ np.linalg.inv(init_transform)

    return T_rel, result


def main(args):
    args.path = os.path.expanduser(args.path)
    args.dataset_root = os.path.expanduser(args.dataset_root)

    pred_files = sorted([
        os.path.join(root, f)
        for root, _, files in os.walk(args.path)
        for f in files
        if f.startswith("prediction_scan_") and f.endswith(".txt")
    ])
    if not pred_files:
        print("❌ No prediction_scan_*.txt files found.")
        return

    refined = 0
    skipped = 0

    for f in tqdm(pred_files, desc="ICP refining"):
        num = os.path.splitext(os.path.basename(f))[0].split("_")[-1]
        folder = os.path.dirname(f)
        obj_name = os.path.basename(folder)

        exr_pattern = f"scan_{num}_positions.exr"
        exr_path = None
        for root, _, files in os.walk(args.dataset_root):
            for p in files:
                if p.endswith(exr_pattern):
                    exr_path = os.path.join(root, p)
                    break
        if not exr_path:
            print(f"[!] Missing EXR for scan {num}")
            skipped += 1
            continue

        stl_path = os.path.join(args.dataset_root, obj_name, "bin.stl")
        if not os.path.isfile(stl_path):
            print(f"[!] Missing STL for {obj_name}")
            skipped += 1
            continue

        pred_T = read_transform_file(f)
        out_path = os.path.join(folder, f"icp_scan_{num}.txt")

        try:
            scene = load_pointcloud_from_exr(exr_path, args.voxel)
            model = load_model_from_stl(stl_path, args.voxel)
        except Exception as e:
            print(f"[!] Load failed: {e}")
            skipped += 1
            continue

        # Crop scene around predicted translation (±crop_range mm)
        crop_range = 30.0  # adjust as needed
        pred_center = pred_T[:3, 3]

        bbox_min = pred_center - crop_range
        bbox_max = pred_center + crop_range
        bbox = o3d.geometry.AxisAlignedBoundingBox(bbox_min, bbox_max)
        scene_cropped = scene.crop(bbox)

        if len(scene_cropped.points) < 100:
            print(f"[!] Too few points after crop ({len(scene_cropped.points)}), skipping {num}")
            skipped += 1
            continue


        # Perform ICP refinement
        T_rel, result = run_icp(model, scene_cropped, args.threshold, pred_T, args.iter)

        # Write only relative transform (baseline expects this)
        write_transform_file(out_path, T_rel)

        d_before = result.inlier_rmse if hasattr(result, "inlier_rmse") else 0
        print(f"[{obj_name} {num}] RMS: {d_before:.3f}")
        refined += 1

    print(f"\n✅ ICP refinement completed: {refined} refined, {skipped} skipped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to inference folder (e.g. inference/resnet34)")
    parser.add_argument("--dataset_root", required=True,
                        help="Root folder with EXR + STL files")
    parser.add_argument("--voxel", type=float, default=0.8)
    parser.add_argument("--iter", type=int, default=80)
    parser.add_argument("--threshold", type=float, default=20.0)
    args = parser.parse_args()
    main(args)
