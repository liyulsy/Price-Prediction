import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class TimeSeriesDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32) if labels is not None else None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.data[idx], self.labels[idx]
        else:
            return self.data[idx]

class BaseLoader:
    @staticmethod
    def load_csv_data(file_path, label_col=None, drop_cols=None):
        df = pd.read_csv(file_path)
        if drop_cols:
            df = df.drop(columns=drop_cols)
        if label_col:
            labels = df[label_col].values
            features = df.drop(columns=[label_col]).values
        else:
            features = df.values
            labels = None
        return features, labels

    @staticmethod
    def create_dataloaders(
        file_path, 
        label_col=None, 
        drop_cols=None, 
        batch_size=32, 
        test_size=0.2, 
        shuffle=True,
        random_state=42,
        norm_type=None  # 'standard' or 'minmax' or None
    ):
        features, labels = BaseLoader.load_csv_data(file_path, label_col, drop_cols)
        # 数据归一化/标准化
        scaler = None
        if norm_type == 'standard':
            scaler = StandardScaler()
            features = scaler.fit_transform(features)
        elif norm_type == 'minmax':
            scaler = MinMaxScaler()
            features = scaler.fit_transform(features)
        # 分割
        if labels is not None:
            X_train, X_test, y_train, y_test = train_test_split(
                features, labels, test_size=test_size, random_state=random_state
            )
        else:
            X_train, X_test = train_test_split(
                features, test_size=test_size, random_state=random_state
            )
            y_train = y_test = None

        train_dataset = TimeSeriesDataset(X_train, y_train) if y_train is not None else TimeSeriesDataset(X_train, np.zeros(len(X_train)))
        test_dataset = TimeSeriesDataset(X_test, y_test) if y_test is not None else TimeSeriesDataset(X_test, np.zeros(len(X_test)))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader, scaler  # 返回scaler便于反归一化

if __name__ == "__main__":
    # 示例
    train_loader, test_loader, scaler = BaseLoader.create_dataloaders(
        file_path='datafiles/outputs_1H.csv',
        drop_cols=['date'],
        batch_size=32,
        norm_type='standard'
    )
    for x, y in train_loader:
        print(x.shape, y.shape)
        break