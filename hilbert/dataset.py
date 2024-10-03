import os
import numpy as np
from torch.utils.data import Dataset

class ModelNetDataset(Dataset):
    def __init__(
        self,
        split='train',
        data_root="data/modelnet40",
        num_points=10000,
    ):
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.num_point = num_points
        self.data_list = self.get_data_list()

    def get_data(self, idx):
        data_idx = idx % len(self.data_list)
        data_name = self.data_list[data_idx]
        
        data_shape = "_".join(data_name.split("_")[0:-1])
        data_path = os.path.join(
                self.data_root, data_shape, self.data_list[data_idx] + ".txt"
            )
        data = np.loadtxt(data_path, delimiter=",").astype(np.float32)
        coord, normal = data[:, 0:3], data[:, 3:6]
        name = self.get_data_name(idx)
        return dict(coord=coord, normal=normal, name=name)

    def get_data_list(self):
        assert isinstance(self.split, str)
        split_path = os.path.join(
            self.data_root, "modelnet40_{}.txt".format(self.split)
        )
        data_list = np.loadtxt(split_path, dtype="str")
        return data_list

    def get_data_name(self, idx):
        data_idx = idx % len(self.data_list)
        return self.data_list[data_idx]

    def __getitem__(self, idx):
        data_dict = self.get_data(idx)
        return data_dict

    def __len__(self):
        return len(self.data_list)
