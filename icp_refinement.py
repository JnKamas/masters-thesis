#!/usr/bin/env python3
import os
import argparse
import numpy as np
import open3d as o3d
import cv2
from tqdm import tqdm
import copy


def read_transform_file(file):
    with open(file, 'r') as f:
        P = f.readline().strip().split(' ')
        R = np.array([[float(P[0]), float(P[4]), float(P[8])],
                      [float(P[1]), float(P[5]), float(P[9])],
                      [float(P[2]), float(P[6]), float(P[10])]])
        t = np.array([float(P[12]), float(P[13]), float(P[14])])
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
    return T


def write_transform_file(file, T):
    with open(file, 'w') as f:
        line = (
            f"{T[0,0]}, {T[1,0]}, {T[2,0]}, 0.0, "
            f"{T[0,1]}, {T[1,1]}, {T[2,1]}, 0.0, "
            f"{T[0,2]}, {T[1,2]}, {T[2,2]}, 0.0, "
            f"{T[0,3]}, {T[1,3]}, {T[2,3]}, 1.0"
        )
        f.write(line)


def load_pointcloud_from_exr(exr_path, voxel_size=0.5):
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


def run_icp(source, target, init_transform, max_iter=50, threshold=5.0):
    if args.iter == 0:
        return init_transform  # skip ICP + pointcloud loading to just test if the results are same
    result = o3d.pipelines.registration.registration_icp(
        source, target, threshold, init_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter)
    )
    return result.transformation


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

    refined_count = 0
    skipped_count = 0

    for f in tqdm(pred_files, desc="Running ICP refinement"):
        num = os.path.splitext(os.path.basename(f))[0].split("_")[-1]
        folder = os.path.dirname(f)

        # ---- Locate the EXR file in dataset_root ----
        pattern = f"scan_{num}_positions.exr"
        candidates = []
        for root, _, files in os.walk(args.dataset_root):
            for p in files:
                if p.endswith(pattern):
                    candidates.append(os.path.join(root, p))

        if not candidates:
            print(f"[!] Missing {pattern} anywhere under {args.dataset_root}, skipping.")
            skipped_count += 1
            continue

        exr_path = candidates[0]
        pred_T = read_transform_file(f)
        out_path = os.path.join(folder, f"icp_scan_{num}.txt")

        try:
            target_pcd = load_pointcloud_from_exr(exr_path, args.voxel)
        except Exception as e:
            print(f"[!] Failed to load {exr_path}: {e}")
            skipped_count += 1
            continue

        # transform predicted points and refine
        source_pcd = copy.deepcopy(target_pcd)
        source_pcd.transform(pred_T)        
        refined_T = run_icp(source_pcd, target_pcd, pred_T,
                            max_iter=args.iter, threshold=args.threshold)
        write_transform_file(out_path, refined_T)
        refined_count += 1

    print(f"\n✅ ICP refinement finished!")
    print(f"   Refined: {refined_count} scans")
    print(f"   Skipped: {skipped_count} (missing or invalid EXRs)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to inference folder (e.g., thesis/inference/resnet34)")
    parser.add_argument("--dataset_root", required=True,
                        help="Path to the dataset root that contains scan_XXX_positions.exr files (can be nested)")
    parser.add_argument("--voxel", type=float, default=0.5, help="Downsample voxel size (mm)")
    parser.add_argument("--iter", type=int, default=50, help="Max ICP iterations")
    parser.add_argument("--threshold", type=float, default=5.0, help="Correspondence threshold (mm)")
    args = parser.parse_args()
    main(args)
