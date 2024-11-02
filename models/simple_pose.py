# Simple Model based on A Simple yet effective baseline for 3D Pose Estimation

import torch.nn as nn
import torch.nn.functional as F

class SimplePose(nn.Module):
    def __init__(self, total_joints, total_actions):
        super().__init__()
        self.total_joints = total_joints,
        self.total_actions = total_actions
        self.input_linear = nn.Linear(total_joints * 2, 1024) # 1d input shape is B x 16 x 2
        self.block1 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.block2 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.output_3d_pose_linear = nn.Linear(1024, total_joints * 3) # 3D output shape is B x 16 x 3
        
        self.output_label_linear = nn.Linear(1024, total_actions) # Predict Action Labels
        
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.input_linear(x)
        x = self.block1(x) + x # First Residual connection
        x = self.block2(x) + x # Second Residual Connection
        three_dim_pose_predictions = self.output_3d_pose_linear(x)
        action_label_predictions = self.output_label_linear(x)
        joint_preds = three_dim_pose_predictions.view(x.shape[0], -1, 3)
        action_preds = action_label_predictions
        return joint_preds, action_preds   # 3D output shape is B x 16 x 3
        