import os

import numpy as np
import torch

from data.camera import normalize_screen_coordinates, world_to_camera

def create_2d_data(data_path, dataset):
    keypoints = np.load(data_path, allow_pickle=True)
    keypoints = keypoints['positions_2d'].item()

    for subject in keypoints.keys():
        for action in keypoints[subject]:
            for cam_idx, kps in enumerate(keypoints[subject][action]):
                # Normalize camera frame
                cam = dataset.cameras()[subject][cam_idx]
                kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])
                keypoints[subject][action][cam_idx] = kps

    return keypoints


def read_3d_data(dataset):
    for subject in dataset.subjects():
        for action in dataset[subject].keys():
            anim = dataset[subject][action]

            positions_3d = []
            for cam in anim['cameras']:
                pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                pos_3d[:, :] -= pos_3d[:, :1]  # Remove global offset
                positions_3d.append(pos_3d)
            anim['positions_3d'] = positions_3d

    return dataset

def create_train_test_files(subjects, dataset, keypoints, type, save_path):
    out_poses_3d = []
    out_poses_2d = []
    out_actions = []

    for subject in subjects:
        for action in keypoints[subject].keys():
            poses_2d = keypoints[subject][action]
            for i in range(len(poses_2d)):  # Iterate across cameras
                out_poses_2d.append(poses_2d[i])
                out_actions.append([action.split(' ')[0]] * poses_2d[i].shape[0])

            if 'positions_3d' in dataset[subject][action]:
                poses_3d = dataset[subject][action]['positions_3d']
                assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                for i in range(len(poses_3d)):  # Iterate across cameras
                    out_poses_3d.append(poses_3d[i])

    if len(out_poses_3d) == 0:
        out_poses_3d = None

    #convert to np arrays
    #TODO.x fails here
    out_poses_3d = np.array(out_poses_3d)
    out_poses_2d = np.array(out_poses_2d)
    out_actions = np.array(out_actions)

    # save 3d poses
    pose_3d_file_path = os.path.join(save_path, f"{type}_3d_poses.npy")
    np.save(pose_3d_file_path, out_poses_3d)
    print(f"Saved 3d poses in file {pose_3d_file_path}")

    # save 2d poses
    pose_2d_file_path = os.path.join(save_path, f"{type}_2d_poses.npy")
    np.save(pose_2d_file_path, out_poses_2d)
    print(f"Saved 2d poses in file {pose_2d_file_path}")

    # save actions
    pose_action_file_path = os.path.join(save_path, f"{type}_actions.npy")
    np.save(pose_action_file_path, out_actions)
    print(f"Saved pose actions in file {pose_action_file_path}")

    return out_poses_3d, out_poses_2d, out_actions