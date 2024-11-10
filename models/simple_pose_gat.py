import dgl
import dgl.nn as dglnn
import dgl.sparse as dglsp
import torch
import torch.nn as nn
import torch.nn.functional as F

# Reference: https://github.com/dmlc/dgl/blob/master/notebooks/sparse/graph_transformer.ipynb

class SparseMultiHeadAttention(nn.Module):
    def __init__(self, hidden_size=80, num_heads=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scaling = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, A, h):
        N = len(h)
        # [N, dh, nh]
        q = self.q_proj(h).reshape(N, self.head_dim, self.num_heads)
        # [N, dh, nh]
        k = self.k_proj(h).reshape(N, self.head_dim, self.num_heads)
        # [N, dh, nh]
        v = self.v_proj(h).reshape(N, self.head_dim, self.num_heads)
        # Compute the MultiHeadAttention with Sparse matrix API
        attention = dglsp.bsddmm(A, q, k.transpose(1, 0)) # Sparse [N, N, nh]
        attention = attention.softmax()  # Sparse [N, N, nh]
        out = dglsp.bspmm(attention, v)  # [N, dh, nh]
        return self.out_proj(out.reshape(N, -1))
    
class GraphTransformerLayer(nn.Module):
    def __init__(self, hidden_size=80, num_heads=8, dropout=0.6):
        super().__init__()
        self.multi_head_attention = SparseMultiHeadAttention(hidden_size, num_heads)
        self.batch_norm_1 = nn.BatchNorm1d(hidden_size)
        self.batch_norm_2 = nn.BatchNorm1d(hidden_size)
        self.feed_forward_1 = nn.Linear(hidden_size, hidden_size * 2)
        self.feed_forward_2 = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout_1 = nn.Dropout(dropout)
        
    def forward(self, A, h):
        h1 = h
        h = self.multi_head_attention(A, h)
        h = self.batch_norm_1(h + h1)
        
        h2 = h
        h = self.feed_forward_2(self.dropout_1(F.relu(self.feed_forward_1(h))))
        h = h2 + h
        
        return self.batch_norm_2(h)
        
class SimplePoseGAT(nn.Module):
    # Reducing layers due to memory issue = num_layers=8
    def __init__(self, in_size, out_size, num_classes, hidden_size=80, num_layers=2, num_heads=8, dropout=0.5, k=20):
        super().__init__()
        self.k = k
        self.embedding_h = nn.Linear(in_size, hidden_size)
        self.pos_linear = nn.Linear(k, hidden_size)
        self.layers = nn.ModuleList(
            [GraphTransformerLayer(hidden_size, num_heads) for _ in range(num_layers)]
        )
        self.pose_predictor = nn.Linear(hidden_size, out_size)
        self.pooler = dglnn.SumPooling()
        
        self.action_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, num_classes),
        )
    
    def forward(self, graph, node_features):
        indices = torch.stack(graph.edges())
        N = graph.num_nodes()
        lap_pe = dgl.lap_pe(graph, k=self.k, padding=True).to(node_features.device)
        A = dglsp.spmatrix(indices, shape=(N, N))
        h = self.embedding_h(node_features) + self.pos_linear(lap_pe)
        for layer in self.layers:
            h = layer(A, h)
        pose_3d = h
        h = self.pooler(graph, h)

        return self.pose_predictor(pose_3d), self.action_predictor(h)