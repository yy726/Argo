from enum import Enum, auto
import os

import numpy as np
import pandas as pd

from data.dataset_manager import DatasetType, dataset_manager


class FeatureName(Enum):
    USER_HISTORY_SEQUENCE_FEATURE = auto()


class FeatureStore:
    """
        This is a component to handle the feature extraction for a given entity.

        For now we only support the movie len related feature extraction.
    """
    def __init__(self):
        self.store = {}

        # we use a precompute strategy to populate the feature values and hold them in memory as of now

        # user history sequence feature
        # TODO: refactor this into a common feature to be shared with dataset preparation
        cached_path = dataset_manager.get_dataset(DatasetType.MOVIE_LENS_LATEST_SMALL)

        ratings = pd.read_csv(os.path.join(cached_path, 'ratings.csv'))

        user_history_sequence_feature = (
            ratings.groupby('userId')
            .apply(lambda x: x.sort_values('timestamp')['movieId'].tolist())
            .reset_index(name="history_sequence_feature")
            .rename(columns={'userId': 'user_id'})
        )  # user_id, historySequenceFeature

        user_history_sequence_feature = user_history_sequence_feature.set_index('user_id')['history_sequence_feature'].to_dict()

        self.store[FeatureName.USER_HISTORY_SEQUENCE_FEATURE] = user_history_sequence_feature

    def generate_feature(self, feature_name, user_id):
        return self.store.get(feature_name).get(user_id)


if __name__ == "__main__":
    feature_store = FeatureStore()

    user_history_sequence = feature_store.generate_feature(FeatureName.USER_HISTORY_SEQUENCE_FEATURE,
                                                           1)
    assert user_history_sequence[0] == 1210
    assert user_history_sequence[-1] == 2492
