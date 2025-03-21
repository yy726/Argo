from torch.utils.data import DataLoader

from data.dataset import prepare_movie_len_dataset
from model.din import DeepInterestModel
from trainer.simple_trainer import SimpleTrainer


if __name__ == "__main__":
    train_dataset, eval_dataset = prepare_movie_len_dataset(history_seq_length=8)
    model = DeepInterestModel()

    trainer = SimpleTrainer(model=model, train_dataset=train_dataset, eval_dataset=eval_dataset)
    trainer.train(num_epochs=5)