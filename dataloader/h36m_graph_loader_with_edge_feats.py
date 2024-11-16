# Convert Human 3.6M dataset to GNN friendly input
from dgl.data import DGLDataset
import numpy as np
import dgl
import torch


class Human36MGraphEdgeDataset(DGLDataset):
    def __init__(self, two_dim_dataset_path, three_dim_dataset_path, label_dataset_path):
        self.two_dim_dataset_path = two_dim_dataset_path
        self.three_dim_dataset_path = three_dim_dataset_path
        self.label_dataset_path = label_dataset_path
        self.two_dim_data = None
        self.three_dim_data = None
        self.labels = None
        self.unique_labels = None
        super().__init__(name="human_3.6m")

    def process(self):
        # Datasets
        self.two_dim_data = np.load(self.two_dim_dataset_path)
        self.three_dim_data = np.load(self.three_dim_dataset_path)
        self.labels = np.load(self.label_dataset_path)
        unique_labels, tags = np.unique(self.labels, return_inverse=True)
        self.unique_labels = unique_labels
        self.labels = tags
        assert len(self.two_dim_data) == len(self.three_dim_data) == len(self.labels)

    def __getitem__(self, index):
        # Data
        two_dim_data, three_dim_data, label = self.two_dim_data[index], self.three_dim_data[index], self.labels[index]
        # Step 1: Define Graph
        # Edge Connections [Source & Destination] <-- Human Body Structure
        human_pose_edge_src = torch.LongTensor([0, 0, 0, 1, 2, 4, 5, 7, 8, 8, 8, 10, 11, 13, 14])
        human_pose_edge_dst = torch.LongTensor([1, 4, 7, 2, 3, 5, 6, 8, 10, 13, 9, 11, 12, 14, 15])
        if 'custom' in self.label_dataset_path:
            human_pose_edge_src = torch.LongTensor([0, 0, 1, 1, 3, 2, 4, 1, 2, 7, 7, 9, 8, 10])
            human_pose_edge_dst = torch.LongTensor([1, 2, 2, 3, 5, 4, 6, 7, 8, 8, 9, 11, 10, 12])
        graph = dgl.graph((human_pose_edge_src, human_pose_edge_dst))
        graph = dgl.to_bidirected(graph)

        # Add node features
        graph.ndata['feat_2d'] = torch.Tensor(two_dim_data)
        graph.ndata['label'] = torch.Tensor(three_dim_data)

        # Add edge features
        src_pos = graph.ndata['feat_2d'][human_pose_edge_src]  # (x1, y1) for each edge
        dst_pos = graph.ndata['feat_2d'][human_pose_edge_dst]  # (x2, y2) for each edge

        # Forward edge features
        forward_features = torch.cat([src_pos, dst_pos], dim=1)  # (x1, y1, x2, y2)

        # Reverse edge features
        reverse_features = torch.cat([dst_pos, src_pos], dim=1)  # (x2, y2, x1, y1)

        # Combine forward and reverse features
        # First n/2 are forward edges, next n/2 are reverse edges
        edge_features = torch.cat([forward_features, reverse_features], dim=0)

        graph.edata['feat'] = edge_features  # Assign to graph's edge features

        return [graph, label]

    def __len__(self):
        return len(self.two_dim_data)
