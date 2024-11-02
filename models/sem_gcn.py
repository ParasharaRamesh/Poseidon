import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

class SemGraphConv(nn.Module):
    """
    Semantic Graph Convolutional Layer used in SemGCN
    """
    def __init__(self, input_features, output_features, graph):
        super(SemGraphConv, self).__init__()
        # Declare and Intialize Layer Weights
        self.weights = nn.Parameter(torch.zeros((2, input_features, output_features)))
        nn.init.xavier_uniform_(self.weights.data, gain=1.414)

        # Declare Bias
        self.bias = nn.Parameter(torch.zeros(output_features))

    def forward(self, graph, h):
        """
        graph: Input Graph
        h: Input Node feature
        """
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.edata['w'] = self.weights
            graph.update_all(message_func=dgl.function.u_mul_e('h', 'w', 'm'), reduce_func=dgl.function.mean('m', 'h_N'))
            h_N = graph.ndata['h_N']
            h_total = torch.cat([h, h_N], dim=1)
            return h_total

class SemGCN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size, num_classes):
        super(SemGCN, self).__init__()
