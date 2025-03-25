import os

import numpy as np
import pandas as pd

from data.dataset_manager import DatasetType, dataset_manager


class RetrievalEngine:

    # TODO: remove `movie_index`
    def __init__(self, movie_index):
        self.movie_index = movie_index

        cached_path = dataset_manager.get_dataset(DatasetType.MOVIE_LENS_LATEST_SMALL)
        movies = pd.read_csv(os.path.join(cached_path, 'movies.csv'))
        # since we have reindex movie during training, we need to use the same reindex map to reindex
        # the movie id, otherwise we would match to the wrong candidates
        movies['movieId'] = pd.Categorical(movies['movieId'], categories=movie_index).codes.astype(np.int64)
        # return each row in a record with column as key name
        self.candidates = movies.to_dict('records')

    def generate_candidates(self):
        """
            Generate candidates based on different retrieval strategy, for now we
            only support retrieval-all strategy. This is seldom used in industry due
            to the high cost. I will add more strategy in the future.
        """
        return [c['movieId'] for c in self.candidates]