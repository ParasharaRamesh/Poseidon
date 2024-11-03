import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import math

class SemGraphConv(nn.Module):
    def __init__(self, input_features, output_features):
        super(SemGraphConv, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_features = input_features
        self.output_features = output_features

        self.e = None

        self.weight = nn.Parameter(torch.zeros(size=(2, input_features, output_features), device=self.device))
        nn.init.xavier_uniform_(self.weight.data, gain=1.414)

        self.bias = nn.Parameter(torch.zeros(output_features, device=self.device))
        std_v = 1.0 / math.sqrt(self.weight.size(2))
        self.bias.data.uniform_(-std_v, std_v)

    def forward(self, graph, h):
        with graph.local_scope():
            # Prepare h0 and h1
            graph.ndata['h0'] = torch.matmul(h, self.weight[0]).to(h.device)
            graph.ndata['h1'] = torch.matmul(h, self.weight[1]).to(h.device)

            # Update edges and apply softmax
            graph.edata['e'] = graph.edata['feat'].to(h.device)
            graph.edata['e'] = F.softmax(graph.edata['e']).to(h.device)

            # Message Passing for h0
            graph.update_all(dgl.function.u_mul_e('h0', 'e', 'm'), dgl.function.sum('m', 'h0_output'))
            graph.update_all(dgl.function.u_mul_e('h1', 'e', 'm'), dgl.function.sum('m', 'h1_output'))

            output = graph.ndata['h0_output'].to(h.device) + graph.ndata['h1_output'].to(h.device) + self.bias.to(h.device)

            return output
            
class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GraphConv, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.graph_conv = SemGraphConv(input_dim, output_dim)
        self.batch_norm = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()

    def forward(self, graph, h):
        h = self.graph_conv(graph, h).to(self.device)
        h = self.batch_norm(h).to(self.device)
        h = self.relu(h).to(self.device)
        return h
    
class ResidualGraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(ResidualGraphConv, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gconv1 = GraphConv(input_dim, hidden_dim)
        self.gconv2 = GraphConv(hidden_dim, output_dim)

    def forward(self, graph, h):
        residual = h
        output = self.gconv1(graph, h).to(self.device)
        output = self.gconv2(graph, output).to(self.device)
        return residual + output

class SemGCN(nn.Module):
    def __init__(self, input_dim, output_dim, hid_dim, num_layers=4, num_classes=15):
        super(SemGCN, self).__init__()
        self.input_layer = GraphConv(input_dim, hid_dim)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.residual_layers = []
        for _ in range(num_layers):
            self.residual_layers.append(ResidualGraphConv(hid_dim, hid_dim, hid_dim))
        
        self.output_layer = SemGraphConv(hid_dim, output_dim)
        self.classifier = nn.Linear(hid_dim, num_classes)
    
    def forward(self, graph, node_features):
        h = self.input_layer(graph, node_features).to(self.device)

        for residual_layer in self.residual_layers:
            h = residual_layer(graph, h).to(self.device)

        output = self.output_layer(graph, h).to(self.device)
        graph.ndata['h'] = h.to(self.device)
        y = dgl.mean_nodes(graph, 'h')
        label = self.classifier(y).to(self.device)
        return output, label