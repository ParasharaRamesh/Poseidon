import dgl
import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F

# Simple GNN Model
class SimplePoseGNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size, num_classes):
        super(SimplePoseGNN, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_size)

        self.conv_1 = dglnn.GraphConv(hidden_size, hidden_size)

        self.block_1 = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.6),
        )

        self.conv_2 = dglnn.GraphConv(hidden_size, hidden_size)

        self.block_2 = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.6),
        )

        self.output_layer_3d = nn.Linear(hidden_size, output_dim)

        self.classification_input = nn.Linear(output_dim, hidden_size)
        self.block_3 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.7),
        )
        self.classifier = nn.Linear(hidden_size, num_classes)
        
    def forward(self, graph, node_features):
        # 3D Pose Estimation
        x = self.embedding(node_features)
        h = self.conv_1(graph, x)
        h = x + self.block_1(h)
        x = h
        h = self.conv_2(graph, h)
        h = x + self.block_2(h)
        h = self.output_layer_3d(h)
        # Classifier
        # Perform classification
        graph.ndata['h'] = h
        x = dgl.mean_nodes(graph, 'h')
        x = self.classification_input(x)
        x = self.block_3(x)
        label = self.classifier(x)
        return h, label