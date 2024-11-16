import dgl
import dgl.nn as dglnn
import torch.nn as nn
import torch
import torch.nn.functional as F
from dgl.nn.pytorch import Sequential

# Reference: https://docs.dgl.ai/en/1.1.x/generated/dgl.nn.pytorch.conv.TAGConv.html

class TAGConvModule(nn.Module):
    def __init__(self, hidden_size, dropout):
        super(TAGConvModule, self).__init__()
        self.conv_1 = dglnn.TAGConv(hidden_size, hidden_size, k=5)
        self.batch_norm_1 = nn.BatchNorm1d(hidden_size)
        
        self.conv_2 = dglnn.TAGConv(hidden_size, hidden_size, k=5)
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
class SimplePoseTAG(nn.Module):
    def __init__(self, hidden_size, num_classes, num_layers=6, dropout=0.5, k=5):
        super(SimplePoseTAG, self).__init__()
        self.k = k
        
        self.input_layer = nn.Linear(2, hidden_size) # B x 16 x 2
        self.pos_linear = nn.Linear(k, hidden_size)
        self.edge_linear = nn.Linear(4, hidden_size)
        
        self.blocks = nn.ModuleList(Sequential(
            TAGConvModule(hidden_size, dropout),
            TAGConvModule(hidden_size, dropout),
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
        
    def forward(self, graph, node_features = None, mode = None):
        lap_pe = dgl.lap_pe(graph, k=self.k, padding=True).to(node_features.device)
        # Compute Laplacian positional encodings
        # Transform edge features
        edge_features = graph.edata['feat']
        processed_edge_features = self.edge_linear(edge_features)  # Shape: (Batch Size x 30, Hidden Size)

        # Aggregate edge features into node features
        graph.edata['processed_feat'] = processed_edge_features
        graph.update_all(
            dgl.function.copy_e('processed_feat', 'm'),  # Send edge features as messages
            dgl.function.sum('m', 'agg_edge_feat')  # Aggregate messages into nodes
        )
        agg_edge_features = graph.ndata['agg_edge_feat']  # Shape: (Batch Size x 16, Hidden Size)

        # Combine node features, positional encodings, and aggregated edge features
        h = self.input_layer(node_features) + self.pos_linear(lap_pe) + agg_edge_features

        for block in self.blocks:
            h = block(graph, h)
        pose_3d_estimations = self.output_3d_pose_linear(h)
            
        graph.ndata['h'] = h
        y = dgl.mean_nodes(graph, 'h')
        label_predictions = self.output_label_linear(y)
        return pose_3d_estimations, label_predictions