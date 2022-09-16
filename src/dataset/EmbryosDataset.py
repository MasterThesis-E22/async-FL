import os
from enum import Enum
from typing import Callable, Tuple, Any, Optional

import numpy as np
import pandas as pd
import torch
import torchvision.datasets.mnist
from PIL import Image as Image
from torchvision.datasets import VisionDataset
from torchvision.io import read_image

class DatasetType(Enum):
    train = 0
    validation = 1
    test = 2

class EmbryosDataset(VisionDataset):

    @property
    def train_labels(self):
        return self.targets

    @property
    def test_labels(self):
        return self.targets

    @property
    def validation_labels(self):
        return self.targets

    @property
    def train_data(self):
        return self.data

    @property
    def test_data(self):
        return self.data

    @property
    def validation_data(self):
        return self.data

    def __init__(
        self,
        root: str,
        dataSetType: DatasetType,
        local: bool = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        percentage_amount_to_include = 1

    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        if local:
            self.data, self.targets = (1-0) * torch.rand((100, 250, 250, 1)) + 0, torch.ones(100, dtype=int)
            return
        self.type = dataSetType
        self.root = root

        # Load metadata
        metadata_file_path = os.path.join(root, "metadata.csv")
        self.meta_data = pd.read_csv(metadata_file_path)

        # For easing loading time - enable possibility of not loading entire dataset
        meta_data_len = len(self.meta_data)
        self.meta_data = self.meta_data.sample(n=int(meta_data_len*percentage_amount_to_include), random_state=42)

        # Split in train and validation
        meta_data_train_validation = self.meta_data.loc[self.meta_data['Testset'] == 0]

        train_size = int(0.8 * len(meta_data_train_validation))
        validation_size = len(meta_data_train_validation) - train_size

        generator = torch.Generator()
        generator.manual_seed(42)
        self.meta_data_train, self.meta_data_validation = torch.utils.data.random_split(meta_data_train_validation, [train_size, validation_size], generator=generator)

        # Load test data
        self.meta_data_test = self.meta_data.loc[self.meta_data['Testset'] == 1]

        self.data, self.targets = self._load_data()

    def _load_data(self):
        batch_size = 10
        data = []

        if self.type == DatasetType.train:
            for index, row in self.meta_data_train.dataset.iterrows():
                try:
                    file_path = os.path.join(self.root, "{:05d}.npz".format(row['SampleID']))
                    img = self._load_image(file_path)
                    data.insert(index, img)
                except:
                    print(f"Cannot load id: {id}" )
                    self.meta_data_train.dataset.drop(index=index)
            label_tensor = torch.LongTensor(self.meta_data_train.dataset["Label"].tolist())
            clinic_ids = np.array(self.meta_data_train.dataset["LabID"].tolist())

        elif self.type == DatasetType.validation:
            for index, row in self.meta_data_validation.dataset.iterrows():
                try:
                    file_path = os.path.join(self.root, "{:05d}.npz".format(row['SampleID']))
                    img = self._load_image(file_path)
                    data.insert(index, img)
                except:
                    print(f"Cannot load id: {id}")
                    self.meta_data_validation.dataset.drop(index=index)
            label_tensor = torch.LongTensor(self.meta_data_validation.dataset["Label"].tolist())
            clinic_ids = np.array(self.meta_data_validation.dataset["LabID"].tolist())

        elif self.type == DatasetType.test:
            for index, row in self.meta_data_test.iterrows():
                try:
                    file_path = os.path.join(self.root, "{:05d}.npz".format(row['SampleID']))
                    img = self._load_image(file_path)
                    data.insert(index, img)
                except:
                    print(f"Cannot load id: {id}")
                    self.meta_data_test.drop(index=index)
            label_tensor = torch.LongTensor(self.meta_data_test["Label"].tolist())
            clinic_ids = np.array(self.meta_data_test["LabID"].tolist())

        data_numpy = np.array(data)
        data_clinic = np.concatenate(
            (np.array(data), np.broadcast_to(clinic_ids[:, None, None], data_numpy.shape[:-1] + (1,))), axis=-1)
        data_clinic_tensor = torch.FloatTensor(data_clinic)
        return data_clinic_tensor, label_tensor

    def _load_image(self, path):
        file_data = np.load(path)
        images = file_data['images']

        focal = 1
        frame = 0
        img_raw = images[frame, :, :, focal]
        img = Image.fromarray(img_raw)
        newsize = (250, 250)
        img = img.resize(newsize)
        img_raw = np.asarray(img)
        img_raw = img_raw.astype('float32') / 255
        return img_raw

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])
        img = img[:,:-1]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode="L")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

