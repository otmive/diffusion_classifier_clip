import os.path as osp
from torchvision import datasets
from diffusion.utils import DATASET_ROOT, get_classes_templates
from diffusion.dataset.objectnet import ObjectNetBase
from diffusion.dataset.imagenet import ImageNet as ImageNetBase
import pandas as pd
import os 
import glob


class MNIST(datasets.MNIST):
    """Simple subclass to override the property"""
    class_to_idx = {str(i): i for i in range(10)}


def get_target_dataset(name: str, train=False, transform=None, target_transform=None, dataset_path=None, data_split=None):
    """Get the torchvision dataset that we want to use.
    If the dataset doesn't have a class_to_idx attribute, we add it.
    Also add a file-to-class map for evaluation
    """

    if name == "cifar10":
        dataset = datasets.CIFAR10(root=DATASET_ROOT, train=train, transform=transform,
                                   target_transform=target_transform, download=True)
    elif name == "stl10":
        dataset = datasets.STL10(root=DATASET_ROOT, split="train" if train else "test", transform=transform,
                                 target_transform=target_transform, download=True)
        dataset.class_to_idx = {cls: i for i, cls in enumerate(dataset.classes)}
    elif name == "pets":
        dataset = datasets.OxfordIIITPet(root=DATASET_ROOT, split="trainval" if train else "test", transform=transform,
                                         target_transform=target_transform, download=True)

        # lower case every key in the class_to_idx
        dataset.class_to_idx = {k.lower(): v for k, v in dataset.class_to_idx.items()}

        dataset.file_to_class = {f.name.split('.')[0]: l for f, l in zip(dataset._images, dataset._labels)}
    elif name == "flowers":
        dataset = datasets.Flowers102(root=DATASET_ROOT, split="train" if train else "test", transform=transform,
                                      target_transform=target_transform, download=True)
        classes = list(get_classes_templates('flowers')[0].keys())  # in correct order
        dataset.class_to_idx = {cls: i for i, cls in enumerate(classes)}

        dataset.file_to_class = {f.name.split('.')[0]: l for f, l in zip(dataset._image_files, dataset._labels)}
    elif name == "aircraft":
        dataset = datasets.FGVCAircraft(root=DATASET_ROOT, split="trainval" if train else "test", transform=transform,
                                        target_transform=target_transform, download=True)

        # replace backslash with underscore -> need to be dirs
        dataset.class_to_idx = {
            k.replace('/', '_'): v
            for k, v in dataset.class_to_idx.items()
        }

        dataset.file_to_class = {
            fn.split("/")[-1].split(".")[0]: lab
            for fn, lab in zip(dataset._image_files, dataset._labels)
        }
        # dataset.file_to_class = {
        #     fn.split("/")[-1].split(".")[0]: lab
        #     for fn, lab in zip(dataset._image_files, dataset._labels)
        # }

    elif name == "food":
        dataset = datasets.Food101(root=DATASET_ROOT, split="train" if train else "test", transform=transform,
                                   target_transform=target_transform, download=True)
        dataset.file_to_class = {
            f.name.split(".")[0]: dataset.class_to_idx[f.parents[0].name]
            for f in dataset._image_files
        }
    elif name == "eurosat":
        if train:
            raise ValueError("EuroSAT does not have a train split.")
        dataset = datasets.EuroSAT(root=DATASET_ROOT, transform=transform, target_transform=target_transform,
                                   download=True)
    elif name == 'imagenet':
        assert not train
        base = ImageNetBase(transform, location=DATASET_ROOT)
        dataset = datasets.ImageFolder(root=osp.join(DATASET_ROOT, 'imagenet/val'), transform=transform,
                                       target_transform=target_transform)
        dataset.class_to_idx = None  # {cls: i for i, cls in enumerate(base.classnames)}
        dataset.classes = base.classnames
        dataset.file_to_class = None
    elif name == 'objectnet':
        base = ObjectNetBase(transform, DATASET_ROOT)
        dataset = base.get_test_dataset()
        dataset.class_to_idx = dataset.label_map
        dataset.file_to_class = None  # todo
    elif name == "caltech101":
        if train:
            raise ValueError("Caltech101 does not have a train split.")
        dataset = datasets.Caltech101(root=DATASET_ROOT, target_type="category", transform=transform,
                                      target_transform=target_transform, download=True)

        dataset.class_to_idx = {cls: i for i, cls in enumerate(dataset.categories)}
        dataset.file_to_class = {str(idx): dataset.y[idx] for idx in range(len(dataset))}
    elif name == "mnist":
        dataset = MNIST(root=DATASET_ROOT, train=train, transform=transform, target_transform=target_transform,
                        download=True)
    elif name == "clevr":
        dataset = datasets.ImageFolder(dataset_path, transform=transform)
        classes = pd.read_csv('../../cobi2_datasets/single_object/single_prompts.csv')
        dataset.class_to_idx = {classes.iloc[i, 1] : classes.iloc[i,2] for i in range(len(classes))}
        # dataset.file_to_class = {}
        # count = 0
        # for name in sorted(glob.glob('/content/drive/MyDrive/single_object/images/*png')):
        #     cur_name = os.path.basename(name)
        #     for i in range(len(dfs)):
        #       filename = dfs.iloc[i,1]
        #       if filename == cur_name:
        #         dataset.file_to_class[str(count)] = dataset.class_to_idx[dfs.iloc[i,2]]
        #         count += 1

    else:
        raise ValueError(f"Dataset {name} not supported.")

    if name in {'mnist', 'cifar10', 'stl10', 'aircraft', 'clevr'}:
        
        train_class_list = {0:1, 1:3, 2:4, 3:5, 4:10, 5:11, 6:13, 7:15, 8:18, 9:19, 10:20, 11:20, 12:23, 13:26, 14:27, 15:28, 16:31 }
        test_class_list = {0:0, 1:2, 2:6, 3:9, 4:12, 5:16, 6:17, 7:21, 8:24, 9:25, 10:30}
        val_class_list = {0:7, 1:8, 2:14, 3:29}
        if data_split == 'train' or data_split == 'id_val' or data_split == 'id_test':
            class_list = train_class_list
        elif data_split == 'ood_val':
            class_list = val_class_list
        elif data_split == 'ood_test':
            class_list = test_class_list

        dataset.file_to_class = {
           str(idx): class_list[dataset[idx][1]]
         for idx in range(len(dataset))
       #     str(idx): dataset[idx][1]
       #     for idx in range(len(dataset))
        }


    print("file to class ", dataset.file_to_class)
    print("class to idx: ", dataset.class_to_idx) # class name : idx
    assert hasattr(dataset, "class_to_idx"), f"Dataset {name} does not have a class_to_idx attribute."
    assert hasattr(dataset, "file_to_class"), f"Dataset {name} does not have a file_to_class attribute."
    return dataset

