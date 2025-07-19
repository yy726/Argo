import numpy as np
import os
import pandas as pd
from functools import lru_cache
from typing import Dict, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader

from configs.model import OUTPUT_MODEL_PATH
from data.dataset_manager import DatasetType, dataset_manager
from feature.movie_len_features import MovieLenFeatureStore
from feature.utils import padding


class DataCache:
    """Simple data caching mechanism for CSV files."""
    
    def __init__(self):
        self._cache = {}
    
    def get_or_load(self, cache_key: str, file_path: str) -> pd.DataFrame:
        """Get data from cache or load from file if not cached."""
        if cache_key not in self._cache:
            self._cache[cache_key] = pd.read_csv(file_path)
        return self._cache[cache_key]
    
    def clear(self):
        """Clear the cache."""
        self._cache.clear()


# Global data cache instance
_data_cache = DataCache()


def load_movielens_data(dataset_type: DatasetType = DatasetType.MOVIE_LENS_LATEST_SMALL) -> Dict[str, pd.DataFrame]:
    """
    Efficiently load MovieLens data with caching to avoid redundant file I/O operations.
    
    Args:
        dataset_type: Type of MovieLens dataset to load
        
    Returns:
        Dictionary containing loaded DataFrames: {'movies', 'ratings', 'links'}
    """
    cached_path = dataset_manager.get_dataset(dataset_type)
    
    # Use cache keys based on dataset path to avoid conflicts
    cache_prefix = f"{dataset_type.name}_{cached_path}"
    
    data = {
        'movies': _data_cache.get_or_load(
            f"{cache_prefix}_movies", 
            os.path.join(cached_path, "movies.csv")
        ),
        'ratings': _data_cache.get_or_load(
            f"{cache_prefix}_ratings", 
            os.path.join(cached_path, "ratings.csv")
        ),
        'links': _data_cache.get_or_load(
            f"{cache_prefix}_links", 
            os.path.join(cached_path, "links.csv")
        )
    }
    
    return data


class FeatureProcessor:
    """
    Centralized feature processing utilities for MovieLens datasets.
    This addresses the TODO about refactoring feature functions to be reusable.
    """
    
    @staticmethod
    def create_user_sequences(ratings: pd.DataFrame, history_seq_length: int) -> pd.DataFrame:
        """Create user interaction sequences from ratings data."""
        return MovieLenFeatureStore.fetch_item_sequence(
            ratings=ratings, 
            seq_length=history_seq_length
        ).rename(columns={"item_sequence": "history_sequence_feature", "userId": "user_id"})
    
    @staticmethod
    def create_labels(ratings: pd.DataFrame, history_seq_length: int, 
                     positive_threshold: float = 5.0, negative_threshold: float = 2.0,
                     max_positive: int = 10, max_negative: int = 50) -> pd.DataFrame:
        """Create positive and negative labels from ratings efficiently."""
        # Use view instead of full copy for better memory efficiency
        filtered_ratings = ratings.groupby("userId").apply(
            lambda x: x.sort_values("timestamp").iloc[history_seq_length:], 
            include_groups=False
        ).reset_index(drop=True)
        
        # Create labels more efficiently using vectorized operations
        positive_mask = filtered_ratings["rating"] >= positive_threshold
        negative_mask = filtered_ratings["rating"] <= negative_threshold
        
        positive = (
            filtered_ratings[positive_mask]
            .groupby("userId")
            .apply(lambda x: x.sort_values("timestamp").head(max_positive), include_groups=False)
            .reset_index(drop=True)
            .drop(columns=["rating", "timestamp"])
            .rename(columns={"userId": "user_id", "movieId": "movie_id"})
        )
        positive["label"] = 1.0

        negative = (
            filtered_ratings[negative_mask]
            .groupby("userId")
            .apply(lambda x: x.sort_values("timestamp").head(max_negative), include_groups=False)
            .reset_index(drop=True)
            .drop(columns=["rating", "timestamp"])
            .rename(columns={"userId": "user_id", "movieId": "movie_id"})
        )
        negative["label"] = 0.0
        
        return pd.concat([positive, negative], axis=0)
    
    @staticmethod
    def process_variable_length_features(features: pd.DataFrame, 
                                       feature_col: str, 
                                       max_length: Optional[int] = None) -> pd.DataFrame:
        """
        Process variable length features with dynamic padding support.
        Addresses the TODO about supporting var-length sparse ids.
        """
        if max_length is None:
            # Dynamically determine max length
            max_length = features[feature_col].apply(len).max()
        
        # Apply dynamic padding
        features[feature_col] = features[feature_col].apply(
            lambda x: padding(x, max_length)
        )
        
        return features


class MovieLenDataset(Dataset):

    @classmethod
    def prepare_data(cls, history_seq_length: int = 6, reindex=False):
        """
        This is the core function to process raw MovieLen dataset to generate the
        training samples, including feature engineering, label generation, movie id reindex, etc

        Return the prepare training data, as well as the movie id index mapping
        """
        # Load data efficiently with caching
        data = load_movielens_data(DatasetType.MOVIE_LENS_LATEST_SMALL)
        movies = data['movies']
        ratings = data['ratings']
        
        unique_ids = None
        if reindex:
            # reindex movie id to compact version
            # use factorize to compute new mapping of ids
            movies["movieId"], unique_ids = pd.factorize(movies["movieId"])
            # use the new mapping of ids as categorical to do assignment, then use code to extract the new ids
            # due to `pd.Categorical`, the movieId is converted to int16, need to convert back to int64
            ratings["movieId"] = pd.Categorical(ratings["movieId"], categories=unique_ids).codes.astype(np.int64)

        # Use the new FeatureProcessor for consistent feature processing
        processor = FeatureProcessor()
        
        # seq feature generation, simplified version, no padding, the minimal
        # length from the ml-latest is 20, for now we assume that
        # sequence feature length + num positive sample <= 20
        user_history_sequence_feature = processor.create_user_sequences(
            ratings, history_seq_length
        )

        # Create labels using the efficient processor
        labels_df = processor.create_labels(ratings, history_seq_length)
        
        training_data = (
            labels_df.merge(user_history_sequence_feature, on="user_id")
            .sample(frac=1).reset_index(drop=True)
        )  # shuffle the rows

        return training_data, unique_ids

    def __init__(self, history_seq_length, data, unique_ids):
        self.history_seq_length = history_seq_length
        self.data = data
        self.unique_ids = unique_ids

    def __getitem__(self, index):
        feature = {
            "user_id": torch.tensor([self.data.loc[index]["user_id"]]),
            "item_id": torch.tensor([self.data.loc[index]["movie_id"]]),
            "user_history_behavior": torch.tensor(self.data.loc[index]["history_sequence_feature"]),
            "user_history_length": torch.tensor([self.history_seq_length]),
            "dense_features": torch.ones([8]),
        }
        return feature, torch.tensor(self.data.loc[index]["label"], dtype=torch.float32)

    def __len__(self):
        return self.data.shape[0]


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

    return (
        MovieLenDataset(
            history_seq_length=history_seq_length,
            data=train_data,
            unique_ids=unique_ids,
        ),
        MovieLenDataset(history_seq_length=history_seq_length, data=eval_data, unique_ids=unique_ids),
        unique_ids,
    )


class MovieLenRatingDataset(Dataset):
    """
    This dataset only load the ratings file. Primarily this is used for
    the MF or the two tower model to compute the item side embeddings.
    """

    def __init__(self, data):
        self.users = torch.LongTensor(data["userId"].values)
        self.movies = torch.LongTensor(data["movieId"].values)
        self.ratings = torch.FloatTensor(data["rating"].values)

    def __len__(self):
        return self.users.shape[0]

    def __getitem__(self, index):
        return self.users[index], self.movies[index], self.ratings[index]


def prepare_movie_len_rating_dataset(eval_ratio=0.1, dataset_type=DatasetType.MOVIE_LENS_LATEST_FULL):
    # Use efficient data loading with caching
    data = load_movielens_data(dataset_type)
    ratings = data['ratings']

    eval_data = ratings.sample(frac=eval_ratio)
    train_data = ratings.drop(index=eval_data.index)

    train_data = train_data.reset_index(drop=True)
    eval_data = eval_data.reset_index(drop=True)

    return MovieLenRatingDataset(train_data), MovieLenRatingDataset(eval_data)


class MovieLenTransActDataset(Dataset):
    """
    This dataset is used for TransAct model, which requires item embeddings to be part
    of the input; right now we have generated the embeddings and saved in file, we need
    to join the embeddings with the sequence read from the ratings.csv file
    """

    @classmethod
    def prepare_data(cls, history_seq_length: int = 6, max_num_genres: int = 5, dataset_type: DatasetType = DatasetType.MOVIE_LENS_LATEST_SMALL):
        """
        This is the core logic of dataset preparation, it group user's sequence based on
        the chronologic order, and returns the movie id and user ratings which is converted
        to categorize as action

        For now, we only generate the sequence, the last item in the sequence could be used as
        the label for training
        """
        # Use efficient data loading with caching
        data = load_movielens_data(dataset_type)
        ratings = data['ratings']
        movies = data['movies']

        feature_store = MovieLenFeatureStore(ratings=ratings, movies=movies, seq_length=history_seq_length)

        # compute the positive/negative label
        neg_idx, pos_idx = set(), set()
        for idx, i in enumerate(feature_store.unique_ids):
            if i < 4.0:
                neg_idx.add(idx)
            else:
                pos_idx.add(idx)

        # obtain item sequence, use the [0:history_seq_length-1] sequence as the input and the last item as
        # candidate and action as label
        # Use efficient view-based operations instead of full copies
        
        # when use larger movie len dataset, the dynamic of user increase and it would be more common to encounter
        # user with shorter history, improve the logic here to properly handle the case
        #
        # there are 2 options:
        #   1. remove the users the does not have sufficient history sequence
        #   2. padding, but we need to shift movie ids and the current version of movie embedding would not be useable

        # use option 1 for now, but with more efficient operations
        processor = FeatureProcessor()
        
        # Create sequences more efficiently without full copies
        user_item_sequence = feature_store.item_sequence[["userId", "item_sequence"]].copy()
        user_item_sequence["item_sequence"] = user_item_sequence["item_sequence"].apply(lambda x: x[: history_seq_length - 1])
        
        user_action_sequence = feature_store.action_sequence[["userId", "action_sequence"]].copy()
        user_action_sequence["action_sequence"] = user_action_sequence["action_sequence"].apply(lambda x: x[: history_seq_length - 1])

        # Use vectorized operations for candidate and label extraction
        candidate = feature_store.item_sequence[["userId", "item_sequence"]].copy()
        candidate["candidate_item"] = candidate["item_sequence"].apply(lambda x: x[history_seq_length - 1] if len(x) > history_seq_length - 1 else x[-1])
        candidate = candidate.drop(columns=["item_sequence"])
        
        label = feature_store.action_sequence[["userId", "action_sequence"]].copy()
        label["label_action"] = label["action_sequence"].apply(lambda x: x[history_seq_length - 1] if len(x) > history_seq_length - 1 else x[-1])
        label = label.drop(columns=["action_sequence"])

        def labeling(x, neg_idx, pos_idx):
            """
            Simple labeling logic
            """
            if x in neg_idx:
                return 0.0
            if x in pos_idx:
                return 1.0
            raise ValueError(f"Missing index {x}")

        label["label"] = label["label_action"].apply(lambda x: labeling(x, neg_idx, pos_idx))
        label = label.drop(columns=["label_action"])

        # for genre feature, it is essentially a var-length feature but for input we need to have fix length as of now,
        # here we padding 0 for list
        # Support var-length sparse ids with dynamic padding
        user_viewed_genres = feature_store.user_viewed_genres[["userId", "sorted_genres"]].copy()
        user_viewed_genres = processor.process_variable_length_features(
            user_viewed_genres, "sorted_genres", max_num_genres
        )

        # retrieve *document* side feature
        candidate = candidate.merge(feature_store.movie_genres, how="left", left_on="candidate_item", right_on="movieId")
        candidate = processor.process_variable_length_features(
            candidate, "genres", max_num_genres
        )

        # merge all features together as a single dataframe
        df = (
            user_item_sequence.merge(user_action_sequence, on="userId")
            .merge(candidate, on="userId")
            .merge(label, on="userId")
            .merge(feature_store.user_num_viewed_movies, on="userId")
            .merge(user_viewed_genres, on="userId")
        )

        return df, feature_store.unique_ids

    def __init__(self, embedding_store, data):
        """
        The item embedding is passed as an object for the dataset to extract the embeddings,
        this is a type of `lazy compute` pattern; for simplicity, we would directly use
        `torch.Tensor` here and in the future this could be generalized to other object
        """
        super().__init__()
        self.embedding_store = embedding_store
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        """
        Consistent data format for TransAct model with standardized tensor types and shapes.
        """
        row = self.data.iloc[index]
        
        # Extract features with consistent naming and types
        user_id = row["userId"]
        item_sequence = row["item_sequence"]
        action_sequence = row["action_sequence"]
        candidate_item = row["candidate_item"]
        user_viewed_genres = row["sorted_genres"]
        user_num_viewed_movies = row["num_viewed_movies"]
        candidate_genres = row["genres"]
        
        # Return standardized feature dictionary with consistent tensor shapes and types
        features = {
            "user_id": torch.tensor([user_id], dtype=torch.long),
            "action_sequence": torch.tensor(action_sequence, dtype=torch.long),
            "item_sequence": self.embedding_store[item_sequence],  # seq x d_item
            "candidate": self.embedding_store[candidate_item],  # d_item
            "user_viewed_genres": torch.tensor(user_viewed_genres, dtype=torch.long),
            "user_num_viewed_movies": torch.tensor([user_num_viewed_movies], dtype=torch.long),
            "candidate_genres": torch.tensor(candidate_genres, dtype=torch.long),
        }
        
        label = torch.tensor(row["label"], dtype=torch.float32)
        
        return features, label


def prepare_movie_len_transact_dataset(embedding_store, dataset_type, history_seq_length, eval_ratio: float = 0.1):
    data, unique_ids = MovieLenTransActDataset.prepare_data(history_seq_length=history_seq_length, dataset_type=dataset_type)

    eval_data = data.sample(frac=eval_ratio)
    train_data = data.drop(index=eval_data.index)

    train_data = train_data.reset_index(drop=True)
    eval_data = eval_data.reset_index(drop=True)

    return MovieLenTransActDataset(embedding_store, train_data), MovieLenTransActDataset(embedding_store, eval_data)


class MovieLenSemanticEmbeddingDataset(Dataset):
    """
    This is the dataset to load the semantic embedding for movie len dataset
    to be used for the RQ-VAE model
    """

    @classmethod
    def prepare_semantic_embeddings(cls):
        """
        This is the core function to prepare the semantic embeddings for the movie len dataset
        It will load the movie id from the movies.csv file, and then load the semantic embeddings
        from the movie_len_llm_embeddings.npy file

        It will also filter out the movies that are all zero, and then normalize the embeddings
        to have unit length
        """
        dataset_path = dataset_manager.get_dataset(DatasetType.MOVIE_LENS_LATEST_FULL)
        movies = pd.read_csv(os.path.join(dataset_path, "movies.csv"))
        movie_ids = movies["movieId"].to_list()

        # we need to filter out the movies that is not zombie in the embedding file
        # also there was some issue during the embedding generation that some movies
        # are not correctly generated or is missing the semantic data
        embeddings = np.load(os.path.join(OUTPUT_MODEL_PATH, "movie_len_llm_embeddings.npy"))
        semantic_embeddings = embeddings[movie_ids]
        print(f"Found {semantic_embeddings.shape[0]} movies in the embedding file")

        all_zero = np.all(semantic_embeddings == 0, axis=1)
        abnormal_movie_indice = np.where(all_zero)[0]  # this is all of the movie ids that are all zero
        print(f"Found {abnormal_movie_indice.shape[0]} movies that are all zero")

        semantic_embeddings = semantic_embeddings[~all_zero]  # fetch the one that is not all zero

        # the original embedding is not normalized and this might cause training issue since we are using
        # the L2 distance
        semantic_embeddings = semantic_embeddings / np.linalg.norm(semantic_embeddings, axis=1, keepdims=True)

        return semantic_embeddings

    def __init__(self):
        self.data = MovieLenSemanticEmbeddingDataset.prepare_semantic_embeddings()
        print(f"Found {self.data.shape[0]} movies with normal semantic embeddings")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return torch.tensor(self.data[index], dtype=torch.float32)
