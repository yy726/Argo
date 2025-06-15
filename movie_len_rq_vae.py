import torch
from torch.utils.data import DataLoader

from data.dataset import MovieLenSemanticEmbeddingDataset
from model.rqvae import RQVAE
from trainer.rqvae_trainer import RQVAETrainer


if __name__ == "__main__":
    train_dataset = MovieLenSemanticEmbeddingDataset()
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    model = RQVAE(input_dim=2048, hidden_dim=1024, output_dim=1024, num_quantize=3)
    trainer = RQVAETrainer(model=model, data_loader=train_loader)
    trainer.train(num_epochs=1)

    # Add some simple test
    model.to("cpu")
    model.eval()
    test_data = train_dataset[0:30]
    test_data = test_data.to("cpu")

    _, ind = model(test_data)
    print(ind)