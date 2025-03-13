from dataclasses import dataclass
import os
import pandas as pd

from dataset import DatasetType, dataset_manager


@dataclass
class Dataset:
    raw_data: dict


class MovieLenDatasetLoader:

    FILES = ['links.csv', 'movies.csv', 'ratings.csv', 'tags.csv']
    
    @classmethod
    def process(cls) -> Dataset:
        cached_path = dataset_manager.get_dataset(DatasetType.MOVIE_LENS_LATEST_SMALL)
        
        data = {}
        for filename in cls.FILES:
            df = pd.read_csv(os.path.join(cached_path, filename))
            data[filename.split('.')[0]] = df
        return Dataset(raw_data=data)


if __name__ == "__main__":
    ml_dataset_loader = MovieLenDatasetLoader.process()

    print(ml_dataset_loader)
