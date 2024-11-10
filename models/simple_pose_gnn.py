import dgl
import dgl.nn as dglnn
import torch.nn as nn
import torch
import torch.nn.functional as F
from dgl.nn.pytorch import Sequential

class GraphConvModule(nn.Module):
    def __init__(self, hidden_size, dropout):
        super(GraphConvModule, self).__init__()
        self.conv_1 = dglnn.GraphConv(hidden_size, hidden_size)
        self.batch_norm_1 = nn.BatchNorm1d(hidden_size)
        
        self.conv_2 = dglnn.GraphConv(hidden_size, hidden_size)
        self.batch_norm_2 = nn.BatchNorm1d(hidden_size)
        
        self.feed_forward_1 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, graph, node_features):
        h_in = node_features
        h = self.conv_1(graph, node_features)
        h = self.batch_norm_1(h)
        h = F.relu(h)
        
        h = self.conv_2(graph, h)
        h = self.batch_norm_2(h)
        h = self.dropout(h)
        h = F.relu(h)
        
        h = self.feed_forward_1(h)
        
        h = h + h_in
        return h
    
# Simple GNN Model
class SimplePoseGNN(nn.Module):
    def __init__(self, hidden_size, num_classes, num_layers=4, dropout=0.5, k=20):
        super(SimplePoseGNN, self).__init__()
        self.k = k
        
        self.input_layer_2d = nn.Linear(2 + self.k, hidden_size) # B x 16 x 2
        self.input_layer_3d = nn.Linear(3 + self.k, hidden_size) # B x 16 x 3
        
        self.blocks = nn.ModuleList(Sequential(
            GraphConvModule(hidden_size, dropout),
            GraphConvModule(hidden_size, dropout),
        ) for _ in range(num_layers))
        
        self.output_3d_pose_linear = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_size, 3) # B x 16 x 3
        )
        
        self.output_label_linear = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )
        
    def forward(self, graph, node_2d_features = None, node_3d_features = None , mode = None):
        
        if mode == 'pose':
            lap_pe = dgl.lap_pe(graph, k=self.k, padding=True).to(node_2d_features.device)
            features = torch.cat([node_2d_features, lap_pe], dim=1)
            h = self.input_layer(features)
            for block in self.blocks:
                h = block(graph, h)
            pose_3d_estimations = self.output_3d_pose_linear(h)
            return pose_3d_estimations
        elif mode == 'activity':
            lap_pe = dgl.lap_pe(graph, k=self.k, padding=True).to(node_3d_features.device)
            features = torch.cat([node_3d_features, lap_pe], dim=1)
            h = self.input_layer(features)
            for block in self.blocks:
                h = block(graph, h)
            graph.ndata['h'] = h
            y = dgl.mean_nodes(graph, 'h')
            label_predictions = self.output_label_linear(y)
            return label_predictions
        elif mode == 'test':
            lap_pe = dgl.lap_pe(graph, k=self.k, padding=True).to(node_2d_features.device)
            features = torch.cat([node_2d_features, lap_pe], dim=1)
            h = self.input_layer(features)
            for block in self.blocks:
                h = block(graph, h)
            pose_3d_estimations = self.output_3d_pose_linear(h)
            lap_pe = dgl.lap_pe(graph, k=self.k, padding=True).to(node_3d_features.device)
            features = torch.cat([node_3d_features, lap_pe], dim=1)
            h = self.input_layer(features)
            for block in self.blocks:
                h = block(graph, h)
            graph.ndata['h'] = h
            y = dgl.mean_nodes(graph, 'h')
            label_predictions = self.output_label_linear(y)
            return pose_3d_estimations, label_predictions