from functools import lru_cache

from data.dataset_manager import DatasetType, dataset_manager


class RetrievalEngine:

    # TODO: remove `movie_index`
    def __init__(self, movie_index):
        self.movie_index = movie_index

        #



# use lru cache as of now give we won't generate dynamic candidates yet
@lru_cache
def simple_candidate_generation(movie_index):
    return MovieLenDataset.candidate_generation(movie_index=movie_index)