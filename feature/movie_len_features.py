import hashlib
import pandas as pd
from collections import Counter


"""
    This module contains all the util functions for feature engineering
    of the MovieLen dataset
"""

# Padding constants for sequences
ITEM_PADDING_TOKEN = 0  # Used for item sequences when length is insufficient
ACTION_PADDING_TOKEN = -1  # Used for action sequences when length is insufficient

GENRES = [
    "Adventure",
    "Animation",
    "Children",
    "Comedy",
    "Fantasy",
    "Romance",
    "Drama",
    "Action",
    "Crime",
    "Thriller",
    "Horror",
    "Mystery",
    "Sci-Fi",
    "IMAX",
    "Documentary",
    "War",
    "Musical",
    "Western",
    "Film-Noir",
    "(no genres listed)",
]


class MovieLenFeatureStore:

    def __init__(self, ratings: pd.DataFrame, movies: pd.DataFrame, seq_length: int = 10):
        """
        Prepare features for MovieLen dataset

        For now we would support it on the MovieLen latest dataset, which is
        small enough to be hold in memory; later once we need to process larger
        dataset, we would write to files for training and dedicated feature store
        to host for serving
        """
        self.ratings = ratings
        self.movies = movies
        self.seq_length = seq_length

        # indexed by userId, sort by userId to make the access consistent
        self.item_sequence = MovieLenFeatureStore.fetch_item_sequence(ratings=ratings, seq_length=seq_length).sort_values(by="userId")
        self.action_sequence, self.unique_ids = MovieLenFeatureStore.fetch_action_sequence(ratings=ratings, seq_length=seq_length)
        self.action_sequence = self.action_sequence.sort_values(by="userId")
        self.user_viewed_genres = MovieLenFeatureStore.fetch_user_viewed_genres(ratings=ratings, movies=movies)
        self.user_num_viewed_movies = MovieLenFeatureStore.fetch_num_viewed_movies(ratings=ratings)

        # indexed by movieId
        self.movie_genres = MovieLenFeatureStore.fetch_movie_genres(movies=movies)

    @classmethod
    def _pad_sequence(cls, sequence: list, target_length: int, padding_token):
        """
        Pad a sequence to target length using the specified padding token.
        
        Args:
            sequence: Input sequence to pad
            target_length: Desired length of the sequence
            padding_token: Token to use for padding
            
        Returns:
            Padded sequence of target_length
        """
        if len(sequence) >= target_length:
            return sequence[:target_length]
        else:
            return sequence + [padding_token] * (target_length - len(sequence))

    @classmethod
    def fetch_item_sequence(cls, ratings: pd.DataFrame, seq_length: int = 10, min_seq_length: int = 1):
        """
        A helper function to extract user rated movie ids in chronologic order, from earliest to latest
        
        Now supports padding for sequences shorter than seq_length to preserve more samples.
        Uses ITEM_PADDING_TOKEN (0) for padding insufficient sequences.

        Args:
            ratings: DataFrame with user ratings
            seq_length: Target sequence length
            min_seq_length: Minimum sequence length to keep (users with fewer interactions are dropped)

        Returns:
            DataFrame in the following format: |userId: integer|item_sequence: list[integer]|
        """
        def process_user_sequence(x):
            sequence = x.sort_values("timestamp").head(seq_length)["movieId"].tolist()
            return cls._pad_sequence(sequence, seq_length, ITEM_PADDING_TOKEN)
        
        item_sequence = (
            ratings.groupby("userId")
            .apply(process_user_sequence, include_groups=False)
            .reset_index(name="item_sequence")
        )
        
        # Only drop users with extremely short sequences (less than min_seq_length)
        original_lengths = (
            ratings.groupby("userId")
            .size()
            .reset_index(name="original_length")
        )
        
        item_sequence = item_sequence.merge(original_lengths, on="userId")
        item_sequence = item_sequence[item_sequence["original_length"] >= min_seq_length]
        item_sequence = item_sequence.drop(columns=["original_length"]).reset_index(drop=True)
        
        return item_sequence

    @classmethod
    def fetch_action_sequence(cls, ratings: pd.DataFrame, seq_length: int = 10, min_seq_length: int = 1):
        """
        A helper function to extract user rate as categorical feature, a.k.a an action in traditional recsys
        
        Now supports padding for sequences shorter than seq_length to preserve more samples.
        Uses ACTION_PADDING_TOKEN (-1) for padding insufficient sequences.

        Args:
            ratings: DataFrame with user ratings
            seq_length: Target sequence length  
            min_seq_length: Minimum sequence length to keep (users with fewer interactions are dropped)

        Returns:
            Tuple of (DataFrame, unique_ids) where DataFrame format is: |userId: int|action_sequence: list[int]|
        """
        # convert the ratings to category
        ratings["rating_cat"], unique_ids = pd.factorize(ratings["rating"])

        def process_user_action_sequence(x):
            sequence = x.sort_values("timestamp").head(seq_length)["rating_cat"].tolist()
            return cls._pad_sequence(sequence, seq_length, ACTION_PADDING_TOKEN)
        
        action_sequence = (
            ratings.groupby("userId")
            .apply(process_user_action_sequence, include_groups=False)
            .reset_index(name="action_sequence")
        )
        
        # Only drop users with extremely short sequences (less than min_seq_length)
        original_lengths = (
            ratings.groupby("userId")
            .size()
            .reset_index(name="original_length")
        )
        
        action_sequence = action_sequence.merge(original_lengths, on="userId")
        action_sequence = action_sequence[action_sequence["original_length"] >= min_seq_length]
        action_sequence = action_sequence.drop(columns=["original_length"]).reset_index(drop=True)

        # revert the addition of column to make it side effect free
        ratings.drop(columns=["rating_cat"], inplace=True)

        return action_sequence, unique_ids

    @classmethod
    def fetch_movie_genres(cls, movies: pd.DataFrame):
        """
        A helper function to convert the movie genres into a list of categorical type

        return DataFrame in the following format

            |movieId: int|genres: list[int]|
        """
        # make a copy to avoid change original data, side effect free
        df = movies[["movieId", "genres"]].copy()
        # split and explode each genre into a single row
        df["genres"] = df["genres"].str.split("|")
        df = df.explode("genres")
        # categorical encoding
        df["genres"] = pd.Categorical(values=df["genres"], categories=GENRES).codes
        # reduce, agg implicitly have a `x.tolist()`
        df = df.groupby("movieId")["genres"].agg(lambda x: sorted(list(set(x)))).reset_index()
        return df

    @classmethod
    def fetch_user_viewed_genres(cls, ratings: pd.DataFrame, movies: pd.DataFrame):
        """
        A helper function to extract all genres that a user has viewed

        return DataFrame in the following format

            |user_id: int|genres: list[int]|counts: list[int]|
        """
        movie_genres = cls.fetch_movie_genres(movies=movies)
        df = ratings[["userId", "movieId"]].merge(movie_genres, on="movieId")

        def func(x):
            """
            `x` is the group result from Pandas, each row in `x` is a list of
            the genres the movie user rated
            """
            # Flatten the list of lists and count occurrences
            genre_counts = Counter(g for genres in x for g in genres)
            genre_counts = list(genre_counts.items())
            # sort based on count in reverse order
            genre_counts = sorted(genre_counts, key=lambda x: (-x[1], x[0]))  # sort based on counts, genre id
            return [x[0] for x in genre_counts], [x[1] for x in genre_counts]

        df = df.groupby("userId")["genres"].agg(lambda x: func(x)).reset_index()  # user_id, genres
        df["sorted_genres"] = df["genres"].apply(lambda x: x[0])
        df["count_genres"] = df["genres"].apply(lambda x: x[1])
        df.drop(columns=["genres"], inplace=True)
        return df

    @classmethod
    def fetch_num_viewed_movies(cls, ratings: pd.DataFrame):
        """
        A helper function to retrieve the number of viewed movies for each user, as
        a dense numeric feature

        return DataFrame in the following format

            |user_id: int|num_viewed_movies: float|
        """
        df = ratings.groupby("userId")["movieId"].agg(lambda x: len(x)).reset_index(name="num_viewed_movies")
        return df


if __name__ == "__main__":
    # Test with original data
    movies = pd.DataFrame(
        {
            "movieId": [1, 2, 3],
            "genres": ["Adventure|Animation", "Drama", "Action|Animation"],
        }
    )
    ratings = pd.DataFrame(
        {
            "userId": [1, 1, 1, 2, 2, 3],  # Added user 3 with only 1 rating to test padding
            "movieId": [1, 2, 3, 2, 3, 1],
            "rating": [5.0, 2.0, 3.0, 1.0, 2.0, 4.0],
            "timestamp": [1, 2, 3, 4, 5, 6],
        }
    )

    # Test movie genres functionality
    df = MovieLenFeatureStore.fetch_movie_genres(movies=movies)
    expected_df = pd.DataFrame({"movieId": [1, 2, 3], "genres": [[0, 1], [6], [1, 7]]})
    result = df.merge(expected_df, on="movieId", suffixes=["x", "y"])
    check = result["genresx"] == result["genresy"]
    assert all(check)
    print("MovieLen movie genres feature test pass...")

    # Test item sequence with padding
    seq_length = 3
    item_seq_df = MovieLenFeatureStore.fetch_item_sequence(ratings=ratings, seq_length=seq_length)
    print(f"Item sequences with padding:\n{item_seq_df}")
    
    # Verify that user 3 (with only 1 rating) gets padded
    user_3_seq = item_seq_df[item_seq_df["userId"] == 3]["item_sequence"].iloc[0]
    expected_user_3_seq = [1, ITEM_PADDING_TOKEN, ITEM_PADDING_TOKEN]
    assert user_3_seq == expected_user_3_seq, f"Expected {expected_user_3_seq}, got {user_3_seq}"
    print("Item sequence padding test pass...")

    # Test action sequence with padding  
    action_seq_df, unique_ids = MovieLenFeatureStore.fetch_action_sequence(ratings=ratings, seq_length=seq_length)
    print(f"Action sequences with padding:\n{action_seq_df}")
    
    # Verify that user 3 gets padded with ACTION_PADDING_TOKEN
    user_3_action_seq = action_seq_df[action_seq_df["userId"] == 3]["action_sequence"].iloc[0]
    # First element should be the rating category, followed by padding tokens
    assert len(user_3_action_seq) == seq_length
    assert user_3_action_seq[1] == ACTION_PADDING_TOKEN
    assert user_3_action_seq[2] == ACTION_PADDING_TOKEN
    print("Action sequence padding test pass...")

    # Test user viewed genres functionality
    df = MovieLenFeatureStore.fetch_user_viewed_genres(ratings=ratings, movies=movies)
    print(f"User viewed genres:\n{df}")
    print("MovieLen user viewed genres test pass...")

    # Test num viewed movies functionality
    df = MovieLenFeatureStore.fetch_num_viewed_movies(ratings=ratings)
    expected_df = pd.DataFrame(
        {
            "userId": [1, 2, 3],
            "num_viewed_movies": [3, 2, 1],
        }
    )
    result = df.merge(expected_df, on="userId", suffixes=["_x", "_y"])
    check = result["num_viewed_movies_x"] == result["num_viewed_movies_y"]
    assert all(check)
    print("MovieLen user num viewed movies test pass...")
    
    print("All tests passed! PADDING functionality successfully implemented.")
