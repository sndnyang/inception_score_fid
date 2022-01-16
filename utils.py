import os
import torchvision as tv
import torchvision.transforms as tr
from torch.utils.data import DataLoader, Dataset
import numpy as np


class DataSubset(Dataset):
    def __init__(self, base_dataset, inds=None, size=-1):
        self.base_dataset = base_dataset
        if inds is None:
            inds = np.random.choice(list(range(len(base_dataset))), size, replace=False)
        self.inds = inds

    def __getitem__(self, index):
        base_ind = self.inds[index]
        return self.base_dataset[base_ind]

    def __len__(self):
        return len(self.inds)


def cycle(loader):
    while True:
        for data in loader:
            yield data


def dataset_with_indices(cls):
    """
    Modifies the given Dataset class to return a tuple data, target, index
    instead of just data, target.
    """

    def __getitem__(self, index):
        data, target = cls.__getitem__(self, index)
        return data, target, index

    return type(cls.__name__, (cls,), {
        '__getitem__': __getitem__,
    })


def dataset_fn(args, train, transform):
    if args.dataset == "cifar10":
        args.n_classes = 10
        cls = dataset_with_indices(tv.datasets.CIFAR10)
        return cls(root=args.data_root, transform=transform, download=True, train=train)
    elif args.dataset == "cifar100":
        args.n_classes = 100
        cls = dataset_with_indices(tv.datasets.CIFAR100)
        return cls(root=args.data_root, transform=transform, download=True, train=train)
    elif args.dataset == 'tinyimagenet':
        args.n_classes = 200
        cls = dataset_with_indices(tv.datasets.ImageFolder)
        return cls(root=os.path.join(args.data_root, 'train' if train else 'val'), transform=transform)
    elif 'img' in args.dataset:
        args.n_classes = 1000
        cls = dataset_with_indices(tv.datasets.ImageFolder)
        return cls(root=os.path.join(args.data_root, 'train' if train else 'val'), transform=transform)
    else:
        args.n_classes = 10
        cls = dataset_with_indices(tv.datasets.SVHN)
        return cls(root=args.data_root, transform=transform, download=True, split="train" if train else "test")


def get_train_test(args):
    mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    if args.dataset == 'img256' or args.dataset == 'imagenet':
        transform_px = tr.Compose(
            [
                tr.Resize(256),
                tr.CenterCrop(224),
                tr.ToTensor(),
                tr.Normalize(mean, std),
             ]
        )
    else:
        transform_px = tr.Compose(
            [
                tr.ToTensor(),
                tr.Normalize(mean, std),
             ]
        )
    # get all training indices
    full_train = dataset_fn(args, True, transform_px)
    all_inds = list(range(len(full_train)))
    # set seed
    np.random.seed(args.seed)
    # shuffle
    np.random.shuffle(all_inds)
    train_inds = np.array(all_inds)

    dset_train = DataSubset(dataset_fn(args, True, transform_px), inds=train_inds)
    dset_test = dataset_fn(args, False, transform_px)

    dload_train_labeled = DataLoader(dset_train, batch_size=100, shuffle=True, num_workers=0, drop_last=True)
    dload_test = DataLoader(dset_test, batch_size=100, shuffle=False, num_workers=0, drop_last=False)
    return dload_train_labeled, dload_test
