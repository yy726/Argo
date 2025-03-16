from torch.utils.data import DataLoader

from data.dataset import MovieLenDataset
from model.din import DeepInterestModel


if __name__ == "__main__":
    dataset = MovieLenDataset(history_seq_length=8)
    loader = DataLoader(dataset=dataset, batch_size=16)
    model = DeepInterestModel()

    # TODO: the current version would fail because the movie id is exceeding the hard-coded
    # item embedding table cardinality, the total cardinality in ML small is around 10k,
    # need to reindex the movie-ids to make it more compact
    for feature, label in loader:
        out = model(feature)
        print(out)

        break