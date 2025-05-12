import torch
from torch.utils.data import DataLoader

from configs.model import DEFAULT_TRANSACT_CONFIG
from data.dataset import prepare_movie_len_transact_dataset
from model.transact import TransActModule, TransAct
from trainer.simple_trainer import SimpleTrainer


DEFAULT_MOVIE_LEN_EMBEDDING_PATH = "artifacts/movie_embeddings.pt"

if __name__ == "__main__":
    embedding_store = torch.load(DEFAULT_MOVIE_LEN_EMBEDDING_PATH, weights_only=False)

    model = TransAct(DEFAULT_TRANSACT_CONFIG)
    train_dataset, eval_dataset = prepare_movie_len_transact_dataset(embedding_store=embedding_store)

    trainer = SimpleTrainer(model=model, train_dataset=train_dataset, eval_dataset=eval_dataset)
    trainer.train(num_epochs=5)
