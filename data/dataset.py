import numpy as np
import os
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from data.dataset_manager import DatasetType, dataset_manager


class MovieLenDataset(Dataset):

    @classmethod
    def prepare_data(cls, history_seq_length: int = 6, reindex=False):
        """
            This is the core function to process raw MovieLen dataset to generate the
            training samples, including feature engineering, label generation, movie id reindex, etc

            Return the prepare training data, as well as the movie id index mapping 
        """
        cached_path = dataset_manager.get_dataset(DatasetType.MOVIE_LENS_LATEST_SMALL)

        # TODO optimization to preprocess data and load on-demand
        links = pd.read_csv(os.path.join(cached_path, 'links.csv'))
        movies = pd.read_csv(os.path.join(cached_path, 'movies.csv'))
        ratings = pd.read_csv(os.path.join(cached_path, 'ratings.csv'))
        links = pd.read_csv(os.path.join(cached_path, 'links.csv'))

        unique_ids = None
        if reindex:
            # reindex movie id to compact version
            # use factorize to compute new mapping of ids
            movies['movieId'], unique_ids = pd.factorize(movies['movieId'])
            # use the new mapping of ids as categorical to do assignment, then use code to extract the new ids
            # due to `pd.Categorical`, the movieId is converted to int16, need to convert back to int64
            ratings['movieId'] = pd.Categorical(ratings['movieId'], categories=unique_ids).codes.astype(np.int64)

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
        # negative candidate generation, use rating <= 2.0 as negative
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

        training_data = (
            pd.concat([positive, negative], axis=0)  # user_id, movie_id, label
            .merge(user_history_sequence_feature, on='user_id')
        ).sample(frac=1).reset_index(drop=True)  # shuffle the rows

        return training_data, unique_ids

    def __init__(self, history_seq_length, data, unique_ids):
        self.history_seq_length = history_seq_length
        self.data = data
        self.unique_ids = unique_ids

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        feature = {
            "user_id": torch.tensor([self.data.loc[index]['user_id']]),
            "item_id": torch.tensor([self.data.loc[index]['movie_id']]),
            "user_history_behavior": torch.tensor(self.data.loc[index]['history_sequence_feature']),
            "user_history_length": torch.tensor([self.history_seq_length]),
            "dense_features": torch.ones([8]),
        }
        return feature, torch.tensor(self.data.loc[index]['label'], dtype=torch.float32)


def prepare_movie_len_dataset(history_seq_length: int = 6, eval_ratio: float = 0.2, reindex=False):
    """
        A helper function to prepare movie len dataset, since the MovieLen dataset is not split
        before hand into train/val/eval, I use this function to read the entire dataset and do
        the split of data to avoid potential leakage
    """
    data, unique_ids = MovieLenDataset.prepare_data(history_seq_length=history_seq_length, reindex=reindex)  # this returns all data

    eval_data = data.sample(frac=eval_ratio)
    train_data = data.drop(eval_data.index)

    train_data = train_data.reset_index(drop=True)
    eval_data = eval_data.reset_index(drop=True)

    return MovieLenDataset(history_seq_length=history_seq_length, 
                           data=train_data, 
                           unique_ids=unique_ids), \
           MovieLenDataset(history_seq_length=history_seq_length,
                           data=eval_data,
                           unique_ids=unique_ids), \
           unique_ids 



if __name__ == "__main__":
    train_dataset, eval_dataset, movie_index = prepare_movie_len_dataset(history_seq_length=8, reindex=True)

    loader = DataLoader(dataset=train_dataset, batch_size=5)

    it = iter(loader)
    batch = next(it)

    assert batch[0]['user_id'].shape == (5, 1)
    assert batch[0]['item_id'].shape == (5, 1)
    assert batch[0]['user_history_behavior'].shape == (5, 8)

    assert batch[1].shape == (5,)

    # check type of the output
    assert batch[0]['user_id'].dtype == torch.int64
    assert batch[0]['item_id'].dtype == torch.int64

    print("Batch shape test passed...")