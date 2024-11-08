import dgl
import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import Sequential

class GraphConvModule(nn.Module):
    def __init__(self, hidden_size, dropout):
        super(GraphConvModule, self).__init__()
        self.conv = dglnn.GraphConv(hidden_size, hidden_size)
        self.block = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
    def forward(self, graph, node_features):
        x = self.conv(graph, node_features)
        x = self.block(x)
        return x
        
# Simple GNN Model
class SimplePoseGNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size, num_classes, num_layers=6, dropout=0.6):
        super(SimplePoseGNN, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_size)
        
        self.blocks = nn.ModuleList(Sequential(
            GraphConvModule(hidden_size, dropout),
            GraphConvModule(hidden_size, dropout),
        ) for _ in range(num_layers))
        
        self.output_3d_pose_linear = nn.Linear(hidden_size, output_dim)
        
        self.output_label_linear = nn.Linear(hidden_size, num_classes)
        
    def forward(self, graph, node_features):
        # 3D Pose Estimation
        h = self.input_layer(node_features)
        
        for block in self.blocks:
            h = h + block(graph, h)
        
        # Classifier
        # Perform classification
        pose_3d_estimations = self.output_3d_pose_linear(h)
        graph.ndata['h'] = h
        y = dgl.mean_nodes(graph, 'h')
        label_predictions = self.output_label_linear(y)
        return pose_3d_estimations, label_predictions