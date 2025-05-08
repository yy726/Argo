import pandas as pd

from data.dataset_manager import dataset_manager

"""
    This module contains all the util functions for feature engineering
    of the MovieLen dataset
"""


GENRES = ['Adventure', 'Animation', 'Children', 'Comedy', 'Fantasy',
       'Romance', 'Drama', 'Action', 'Crime', 'Thriller', 'Horror',
       'Mystery', 'Sci-Fi', 'IMAX', 'Documentary', 'War', 'Musical',
       'Western', 'Film-Noir', '(no genres listed)']


class MovieLenFeatureStore:

    def __init__(self, ratings: pd.DataFrame, movies: pd.DataFrame):
        self.ratings = ratings
        self.movies = movies
    
    @classmethod
    def fetch_item_sequence(cls, ratings: pd.DataFrame, seq_length: int = 10):
        """
        A helper function to extract user rated movie ids in chronologic order, from earliest to latest

        return DataFrame in the following format

            |user_id: integer|item_sequence_feature: list[integer]|
        """
        item_sequence = (
            ratings.groupby("userId")
            .apply(lambda x: x.sort_values("timestamp").head(seq_length)["movieId"].tolist())  # for group less than `seq_length`, this would return all rows
            .reset_index(name="item_sequence")
            .rename(columns={"userId": "user_id"})
        )  # user_id, item_sequence_feature
        return item_sequence
    
    @classmethod
    def fetch_action_sequence(cls, ratings: pd.DataFrame, seq_length: int = 10):
        """
        A helper function to extract user rate as categorical feature, a.k.a an action in traditional recsys

        return DataFrame in the following format
        """
         # convert the ratings to category
        ratings["rating_cat"], unique_ids = pd.factorize(ratings["rating"])
        
        action_sequence = (
            ratings.groupby("userId")
            .apply(lambda x: x.sort_values("timestamp").head(seq_length)["rating_cat"].tolist())
            .reset_index(name="action_sequence")
            .rename(columns={"userId": "user_id"})
        )

        # revert the addition of column to make it side effect free
        ratings.drop(columns=["rating_cat"], inplace=True)

        return action_sequence, unique_ids

    @classmethod
    def fetch_movie_genres(cls, movies: pd.DataFrame):
        """
        A helper function to convert the movie genres into a list of categorical type
        """


        