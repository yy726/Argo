from torch.utils.data import DataLoader

from data.dataset import MovieLenDataset
from model.din import DeepInterestModel
from trainer.simple_trainer import SimpleTrainer


if __name__ == "__main__":
    dataset = MovieLenDataset(history_seq_length=8)
    loader = DataLoader(dataset=dataset, batch_size=16)
    model = DeepInterestModel()

    trainer = SimpleTrainer(model=model, data_loader=loader)
    trainer.train(num_epochs=3)