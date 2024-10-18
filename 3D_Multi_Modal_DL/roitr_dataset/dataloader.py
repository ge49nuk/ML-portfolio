import torch
from functools import partial
from roitr_dataset.common import load_info, collate_fn
from roitr_dataset.bop import BOP


def get_dataset(config):
    '''
    Make pytorch dataset for training, validation and testing
    :param config: configuration
    :return: train_set: training dataset
             val_set: validation dataset
             benchmark_set: testing dataset
    '''
    if config.dataset == 'bop':
        training_set = BOP(config, 'train')
        val_set = BOP(config, 'val')
        testing_set = BOP(config, 'test')
    else:
        raise NotImplementedError

    return training_set, val_set, testing_set


def get_dataloader(dataset, sampler=None, batch_size=1, num_workers=8, shuffle=True, drop_last=True):
    '''
    Get the pytorch dataloader for specific pytorch dataset
    :param dataset: pytorch dataset
    :param batch_size: size of a batch of data
    :param num_workers: the number of threads used in dataloader
    :param shuffle: whether to shuffle dataset for each epoch
    :return: pytorch dataloader
    '''
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        num_workers=num_workers,
        collate_fn=partial(collate_fn, config=dataset.config),
        drop_last=drop_last
    )
    return data_loader

