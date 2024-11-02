import dgl
import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F

# Simple GNN Model
class SimplePoseGNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size, num_classes):
        super(SimplePoseGNN, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_size)
        self.conv1 = dglnn.GraphConv(hidden_size, hidden_size)
        self.batch_norm1d = nn.BatchNorm1d(hidden_size)
        self.linear_layer_2 = nn.Linear(hidden_size, hidden_size)
        self.conv2 = dglnn.GraphConv(hidden_size, hidden_size)
        self.batch_norm1d_2 = nn.BatchNorm1d(hidden_size, hidden_size)
        self.linear_layer_3 = nn.Linear(hidden_size, output_dim)
        self.classifier = nn.Linear(output_dim, num_classes)
        
    def forward(self, graph, node_features):
        # Convert Node features into Node Embeddings using Linear Layer
        x = self.embedding(node_features)
        # Perform Graph Convolutions with ReLU
        h = x + self.conv1(graph, x)
        # Batch Norm
        h = self.batch_norm1d(h)
        # Non Linearality
        h = F.relu(h)
        # Linearize
        h = self.linear_layer_2(h)
        # Graph Convolution
        h = x + self.conv2(graph, h)
        # Batch Norm
        h = self.batch_norm1d_2(h)
        # Relu
        h = F.relu(h)
        # Output Layer
        h = self.linear_layer_3(h)
        # Perform classification
        graph.ndata['h'] = h
        y = dgl.mean_nodes(graph, 'h')
        label = self.classifier(y)
        return h, label