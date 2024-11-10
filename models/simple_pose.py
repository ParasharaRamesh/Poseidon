# Simple Model based on A Simple yet effective baseline for 3D Pose Estimation

import torch.nn as nn
import torch.nn.functional as F

class SimplePose(nn.Module):
    def __init__(self, total_joints, total_actions, hidden_size=1024, num_layers=6, dropout=0.6):
        super().__init__()
        self.total_joints = total_joints,
        self.total_actions = total_actions
        self.input_linear_2d = nn.Linear(total_joints * 2, hidden_size) # 1d input shape is B x 16 x 2
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
        
        self.output_3d_pose_linear = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, total_joints * 3)
        ) # 3D output shape is B x 16 x 3
        
        self.output_label_linear = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, total_actions)
        ) # Predict Action Labels
        
    def forward(self, x=None, mode=None):
        x = x.view(x.shape[0], -1)
        x = self.input_linear_2d(x)
        for block in self.blocks:
            x = x + block(x)
        three_dim_pose_predictions = self.output_3d_pose_linear(x)
        joint_preds = three_dim_pose_predictions.view(x.shape[0], -1, 3)
        
        action_preds = self.output_label_linear(x)
        return joint_preds, action_preds