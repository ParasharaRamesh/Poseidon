import dgl
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F

# Simple GNN Model
class SimplePoseGAT(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size, num_classes):
        self.input_dim = input_dim + 2 # incorporate additional positional encoding (x,y)
        self.output_dim = output_dim
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        super(SimplePoseGAT, self).__init__()
        self.gat_layers = nn.ModuleList()

        # two-layer GAT with heads (8,1)
        self.heads = [8,1]
        self.gat_layers.append(
            dglnn.GATConv(
                self.input_dim,
                self.hidden_size,
                self.heads[0],
                feat_drop=0.6,
                attn_drop=0.6,
                activation=F.elu,
            )
        )
        self.gat_layers.append(
            dglnn.GATConv(
                self.hidden_size * self.heads[0],
                self.output_dim,
                self.heads[1],
                feat_drop=0.6,
                attn_drop=0.6,
                activation=None,
            )
        )
        self.classifier = nn.Linear(output_dim, num_classes)


    def forward(self, graph, node_2d_features):
        # Concatenate 2D coordinates as positional embeddings
        pos_embeddings = graph.ndata['feat_2d']  # (x, y) coordinates
        h = torch.cat([node_2d_features, pos_embeddings], dim=1)  # Concatenate along feature dimension

        for i, layer in enumerate(self.gat_layers):
            h = layer(graph, h)
            if i == len(self.gat_layers) - 1:  # last layer
                h = h.mean(1)
            else:  # other layer(s)
                h = h.flatten(1)

        # Perform classification
        graph.ndata['h'] = h
        y = dgl.mean_nodes(graph, 'h')
        label = self.classifier(y)

        #return 3d and label outputs
        return h, label