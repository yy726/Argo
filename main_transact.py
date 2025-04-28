from torch.utils.data import DataLoader

from data.dataset import prepare_movie_len_dataset
from model.transact import TransActModule

"""
    The training scripts of TransAct model, right now only test the TransAct Module with
    the actual movie embeddings. 
"""

if __name__ == "__main__":
    