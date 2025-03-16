from dataclasses import dataclass
import os
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from data.dataset_manager import DatasetType, dataset_manager


class MovieLenDataset(Dataset):

    def __init__(self, history_seq_length: int = 6, include_hard_negative: bool = False):
        cached_path = dataset_manager.get_dataset(DatasetType.MOVIE_LENS_LATEST_SMALL)

        # TODO optimization to preprocess data and load on-demand
        links = pd.read_csv(os.path.join(cached_path, 'links.csv'))
        movies = pd.read_csv(os.path.join(cached_path, 'movies.csv'))
        ratings = pd.read_csv(os.path.join(cached_path, 'ratings.csv'))
        links = pd.read_csv(os.path.join(cached_path, 'links.csv'))

        # seq feature generation, simplified version, no padding, the minimal
        # length from the ml-latest is 20, for now we assume that
        # sequence feature length + num positive sample <= 20
        user_history_sequence_feature = (
            ratings.groupby('userId')
            .apply(lambda x: x.sort_values('timestamp').head(history_seq_length)['movieId'].tolist())  # for group less than `history_seq_length`, this would return all rows
            .reset_index(name="history_sequence_feature")
            .rename(columns={'userId': 'user_id'})
        )  # user_id, historySequenceFeature

        # positive candidate generation, use rating >= 5.0 as positive
        # negative candidate generation, use rateing <= 2.0 as negative
        df = (
            ratings.groupby('userId')
            .apply(lambda x: x.sort_values('timestamp').iloc[history_seq_length:])  # use sequence that is not part of the sequence feature to prevent label leakage
            .reset_index(drop=True)
        )

        positive = (
            df[df['rating'] >= 5.0].groupby('userId')
            .apply(lambda x: x.sort_values('timestamp').head(10))
            .reset_index(drop=True)
            .drop(columns=['rating', 'timestamp'])
            .rename(columns={'userId': 'user_id', 'movieId': 'movie_id'})
        )  # user_id, movie_id
        positive['label'] = 1.0

        negative = (
            df[df['rating'] <= 2.0].groupby('userId')
            .apply(lambda x: x.sort_values('timestamp').head(50))
            .reset_index(drop=True)
            .drop(columns=['rating', 'timestamp'])
            .rename(columns={'userId': 'user_id', 'movieId': 'movie_id'})
        )
        negative['label'] = 0.0

        self.training_data = (
            pd.concat([positive, negative], axis=0)  # user_id, movie_id, label
            .merge(user_history_sequence_feature, on='user_id')
        )

    def __len__(self):
        return self.training_data.shape[0]
    
    def __getitem__(self, index):
        feature = {
            "user_id": torch.tensor([self.training_data.loc[index]['user_id']]),
            "item_id": torch.tensor([self.training_data.loc[index]['movie_id']]),
            "user_history_behavior": torch.tensor(self.training_data.loc[index]['history_sequence_feature'])
        }
        return feature, torch.tensor(self.training_data.loc[index]['label'])


if __name__ == "__main__":
    dataset = MovieLenDataset(history_seq_length=8)

    loader = DataLoader(dataset=dataset, batch_size=5)

    it = iter(loader)
    batch = next(it)

    assert batch[0]['user_id'].shape == (5, 1)
    assert batch[0]['item_id'].shape == (5, 1)
    assert batch[0]['user_history_behavior'].shape == (5, 8)

    assert batch[1].shape == (5,)

    print("Batch shape test passed...")