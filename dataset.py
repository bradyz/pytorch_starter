from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms


def make_transformation(dataset_params, normalize, is_train):
    functions = list()

    if is_train:
        functions.append(transforms.RandomHorizontalFlip())

    functions.append(transforms.ToTensor())

    if normalize:
        functions.append(
                transforms.Normalize(dataset_params.mean, dataset_params.std))

    return transforms.Compose(functions)


def _dataloader_wrapper(
        dataset_params, normalize, is_train, batch_size, data_root, threads):
    augment = make_transformation(dataset_params, normalize, is_train)
    data = dataset_params.dataset(
            root=data_root, train=is_train, download=True, transform=augment)

    return DataLoader(
            data, batch_size=batch_size, shuffle=is_train,
            num_workers=threads)


class CIFAR10(object):
    dataset = torchvision.datasets.CIFAR10
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    in_channels = 3
    size = 32
    num_classes = 10

    def get_data(data_root, is_train, batch_size, normalize=True, threads=2):
        return _dataloader_wrapper(
                CIFAR10, normalize, is_train, batch_size, data_root, threads)
