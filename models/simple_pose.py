# Simple Model based on A Simple yet effective baseline for 3D Pose Estimation

import torch.nn as nn
import torch.nn.functional as F

class SimplePose(nn.Module):
    def __init__(self, total_joints, total_actions, hidden_size=1024, num_layers=6, dropout=0.6):
        super().__init__()
        self.total_joints = total_joints,
        self.total_actions = total_actions
        self.input_linear_2d = nn.Linear(total_joints * 2, hidden_size) # 1d input shape is B x 16 x 2
        self.input_linear_3d = nn.Linear(total_joints * 3, hidden_size) # 1d input shape is B x 16 x 2
        self.blocks = nn.ModuleList(nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        ) for _ in range(num_layers))
        
        self.output_3d_pose_linear = nn.Linear(hidden_size, total_joints * 3) # 3D output shape is B x 16 x 3
        
        self.output_label_linear = nn.Linear(hidden_size, total_actions) # Predict Action Labels
        
    def forward(self, x_2d=None, x_3d=None, mode=None):
        if mode == 'pose':
            x = x_2d.view(x_2d.shape[0], -1)
            x = self.input_linear_2d(x)
            for block in self.blocks:
                x = x + block(x)
            three_dim_pose_predictions = self.output_3d_pose_linear(x)
            joint_preds = three_dim_pose_predictions.view(x.shape[0], -1, 3)
            return joint_preds
        elif mode == 'activity':
            x = x_3d.view(x_3d.shape[0], -1)
            x = self.input_linear_3d(x)
            for block in self.blocks:
                x = x + block(x)
            action_preds = self.output_label_linear(x)
            return action_preds
        elif mode == 'test':
            x_2d = x_2d.view(x_2d.shape[0], -1)
            x_2d = self.input_linear_2d(x_2d)
            for block in self.blocks:
                x_2d = x_2d + block(x_2d)
            three_dim_pose_predictions = self.output_3d_pose_linear(x_2d)
            joint_preds = three_dim_pose_predictions.view(x_2d.shape[0], -1, 3)
            
            x_3d = x_3d.view(x_3d.shape[0], -1)
            x_3d = self.input_linear_3d(x_3d)
            for block in self.blocks:
                x_3d = x_3d + block(x_3d)
            action_preds = self.output_label_linear(x_3d)
            return joint_preds, action_preds
        