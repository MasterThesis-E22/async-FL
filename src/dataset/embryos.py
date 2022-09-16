import torch
from torchvision import datasets, transforms

import utils.IID
from dataset.EmbryosDataset import EmbryosDataset, DatasetType
from utils.Tools import *


class embryos:
    def __init__(self, clients, iid_config):
        train_datasets = EmbryosDataset(root='/mnt/data/mlr_ahj_datasets/vitrolife/dataset/', dataSetType=DatasetType.train,
                                               transform=transforms.ToTensor(), percentage_amount_to_include=0.05)
        test_datasets = EmbryosDataset(root='/mnt/data/mlr_ahj_datasets/vitrolife/dataset/', dataSetType=DatasetType.test,
                                              transform=transforms.ToTensor(), percentage_amount_to_include=0.05)
        validation_datasets = EmbryosDataset(root='/mnt/data/mlr_ahj_datasets/vitrolife/dataset/',
                                       dataSetType=DatasetType.validation,
                                       transform=transforms.ToTensor(), percentage_amount_to_include=0.2)

        train_data = train_datasets.data
        self.raw_data = train_datasets.data
        self.train_labels = train_datasets.targets
        test_data = test_datasets.data[:, :, :-1]
        self.test_datasets = test_datasets
        # 归一化
        self.train_data = train_data.reshape(train_data.shape[0], train_data.shape[1] * train_data.shape[2])
        self.test_data = test_data.reshape(test_data.shape[0], test_data.shape[1] * test_data.shape[2])

        self.train_data_size = train_data.shape[0]
        self.datasets = []
        self.iid_config = iid_config


        if isinstance(iid_config, bool):
            for i in range(clients):
                mask = train_data[:, 0, 250] == i
                client_data = train_data[mask]
                client_data = client_data[:, :, :-1]
                client_labels = self.train_labels[mask]

                self.datasets.append(TensorDataset(torch.tensor(client_data), torch.tensor(client_labels)))
        else:
            print("generating non_iid data...")
            label_config = iid_config['label']
            data_config = iid_config['data']
            utils.IID.generate_non_iid_data(label_config, data_config, self, clients, 0, 10)
        print("data generation process completed")

    def get_test_dataset(self):
        return self.test_datasets

    def get_train_dataset(self):
        return self.datasets

    def get_config(self):
        return self.iid_config
