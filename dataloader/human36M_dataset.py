import numpy as np
from torch.utils.data import Dataset

class Human36MDataset(Dataset):
    def __init__(self, two_d_dataset_path, three_d_dataset_path, label_dataset_path):
        self.two_d_dataset_path = two_d_dataset_path
        self.three_d_dataset_path = three_d_dataset_path
        self.label_dataset_path = label_dataset_path
        self.input_data = np.load(self.two_d_dataset_path)
        self.output_data = np.load(self.three_d_dataset_path)
        self.labels = np.load(self.label_dataset_path)
        unique_labels, tags = np.unique(self.labels, return_inverse=True)
        self.unique_labels = unique_labels
        self.labels = tags
        self.labels_map = dict(zip(range(len(unique_labels)),unique_labels))
        assert len(self.input_data) == len(self.labels) == len(self.output_data)
    
    def get_action_numbers(self):
        return len(self.unique_labels)
    
    def get_joint_numbers(self):
        return self.input_data[0].shape[0]
    
    def __len__(self):
        return len(self.input_data)
    
    def __getitem__(self, index):
        return self.input_data[index], self.output_data[index], self.labels[index]