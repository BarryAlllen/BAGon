from torch.utils.data import Dataset, DataLoader


class DataLoaderProducer:
    def __init__(self):
        pass

    def get_loader(self, dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int, drop_last: bool = False):
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last
        )
        return data_loader
