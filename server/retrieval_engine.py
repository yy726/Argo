import os

import numpy as np
import pandas as pd

from data.dataset_manager import DatasetType, dataset_manager


class RetrievalEngine:

    def __init__(
        self,
    ):

        cached_path = dataset_manager.get_dataset(DatasetType.MOVIE_LENS_LATEST_SMALL)
        movies = pd.read_csv(os.path.join(cached_path, "movies.csv"))
        # return each row in a record with column as key name
        self.candidates = movies.to_dict("records")

    def generate_candidates(self):
        """
        Generate candidates based on different retrieval strategy, for now we
        only support retrieval-all strategy. This is seldom used in industry due
        to the high cost. I will add more strategy in the future.
        """
        return [c["movieId"] for c in self.candidates]
