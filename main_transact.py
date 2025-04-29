import torch
from torch.utils.data import DataLoader

from data.dataset import prepare_movie_len_transact_dataset
from model.transact import TransActModule

"""
    The training scripts of TransAct model, right now only test the TransAct Module with
    the actual movie embeddings. 
"""

DEFAULT_MOVIE_LEN_EMBEDDING_PATH = "artifacts/movie_embeddings.pt"

if __name__ == "__main__":
    embedding_store = torch.load(DEFAULT_MOVIE_LEN_EMBEDDING_PATH, weights_only=False)
    transact = TransActModule().to("cpu")
    train_dataset, eval_dataset = prepare_movie_len_transact_dataset(embedding_store=embedding_store)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    for idx, batch in enumerate(train_dataloader):
        user_id = batch['user_id'].to("cpu")
        action_sequence = batch['action_sequence'][:, :-1].to("cpu")  # the last item would be used as label in the future
        item_sequence = batch['item_sequence'][:, :-1, :].to("cpu")
        candidate = batch['item_sequence'][:, -1, :].to("cpu")

        out = transact(action_sequence=action_sequence,
                       item_sequence=item_sequence,
                       candidate=candidate)
        # total around 17 batch, because the data is grouped on user id, and in small dataset we have about 600 users
        print(f"Current batch id {idx}, current model prediction {out}")
