from PIL import Image
import torch
from torch import nn, optim
import glob
import os
import pandas as pd
import numpy as np
import clip
from torch.utils.data import Dataset, DataLoader, BatchSampler
from sklearn.model_selection import train_test_split
import random
from matplotlib.pyplot import imshow
import argparse
import sys

BATCH_SIZE = 8
EPOCH = 30


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class SingleClevrDataset(Dataset):
    def __init__(self, data, preprocess):
        self.preprocess = preprocess
        self.img_paths = []
        self.captions = []

        classes = pd.read_csv('cobi2_datasets/single_object/single_prompts.csv')

        self.class_to_idx = classes['prompt'].tolist()
        for img_path, captions in data.items():
            for cap in captions:
                self.img_paths.append(img_path)
                self.captions.append(cap)
        self.processed_cache = {}
        for img_path in data:
            self.processed_cache[img_path] = self.preprocess(Image.open(img_path))
        self.img_paths_set = list(data.keys())
        self.path2label = {path: self.class_to_idx.index("a photo of a " + path.split('/')[3]) for path in self.img_paths_set}

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = self.processed_cache[img_path]
        caption = self.captions[idx]
        label = self.path2label[img_path]
        return image, caption, label
    
class TwoObjectClevrDataset(Dataset):
    def __init__(self, data, preprocess):
        self.preprocess = preprocess
        self.img_paths = []
        self.captions = []

        classes = ["a photo of a " + c.split("_")[0] + " " + c.split("_")[1] for c in os.listdir('cobi2_datasets/two_object/train')]
        self.class_to_idx = list(set(classes))
        for img_path, captions in data.items():
            for cap in captions:
                self.img_paths.append(img_path)
                self.captions.append(cap)
        self.processed_cache = {}
        for img_path in data:
            self.processed_cache[img_path] = self.preprocess(Image.open(img_path))
        self.img_paths_set = list(data.keys())
        self.path2label = {path: self.class_to_idx.index("a photo of a " + path.split('/')[4]) for path in self.img_paths_set}
    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = self.processed_cache[img_path]
        caption = self.captions[idx]
        label = self.path2label[img_path]
        return image, caption, label
    
class RelClevrDataset(Dataset):
    def __init__(self, data, preprocess):
        self.preprocess = preprocess
        self.img_paths = []
        self.captions = []

        classes = ["a photo of a " + c for c in os.listdir('cobi2_datasets/relational/train')]
        self.class_to_idx = list(set(classes))
        print(self.class_to_idx)
        for img_path, captions in data.items():
            for cap in captions:
                self.img_paths.append(img_path)
                self.captions.append(cap)
        self.processed_cache = {}
        for img_path in data:
            self.processed_cache[img_path] = self.preprocess(Image.open(img_path))
        self.img_paths_set = list(data.keys())
        print(self.img_paths_set[0])
        self.path2label = {path: self.class_to_idx.index("a photo of a " + path.split('/')[3]) for path in self.img_paths_set}
        print(dict(list(self.path2label.items())[:10]))
    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = self.processed_cache[img_path]
        caption = self.captions[idx]
        label = self.path2label[img_path]
        return image, caption, label
    
class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, labels, n_classes, n_samples):
        self.labels = labels
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size
    
def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()

def finetune(dataset, seed, data_path, save_path, batch_size=BATCH_SIZE, epoch=EPOCH):
    set_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)



    if dataset == 'two_object':
        d_train = {}
        for folder in os.listdir(data_path):
            for subfolder in os.listdir(data_path + '/' + folder):
                for img_path in glob.glob(data_path +'/' + folder + "/" + subfolder + "/*.png"):
                    d_train[img_path] = ["a photo of a " + subfolder]
    else:
        d_train = {}
        for folder in os.listdir(data_path):
            for img_path in glob.glob(data_path + "/" + folder + "/*.png"):
                d_train[img_path] = ["a photo of a " + folder]

    if dataset == 'single':
        train_dataset = SingleClevrDataset(d_train, preprocess)
    elif dataset == 'two_object':
        train_dataset = TwoObjectClevrDataset(d_train, preprocess)
    elif dataset == 'relational':
        train_dataset = RelClevrDataset(d_train, preprocess)
    train_labels = torch.tensor([item[2] for item in train_dataset])
    train_sampler = BalancedBatchSampler(train_labels, BATCH_SIZE, 1)
    train_dataloader = DataLoader(train_dataset, batch_sampler=train_sampler)

    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=5e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_dataloader)*EPOCH)

    best_ep = -1
    best_tr_loss = 1e5
    #import wandb



    #wandb.init(project='train_clip_relational', name='vit-l14')
    for epoch in range(EPOCH):
        print(f"running epoch {epoch}, best train loss {best_tr_loss} after epoch {best_ep}")
        step = 0
        tr_loss = 0
        model.train()
        for batch in train_dataloader:
            step += 1
            optimizer.zero_grad()

            images, texts, _ = batch
            images = images.to(device)
            texts = clip.tokenize(texts).to(device)
    #         print(images.shape, texts.shape)
            logits_per_image, logits_per_text = model(images, texts)
            ground_truth = torch.arange(BATCH_SIZE).to(device)

            total_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
            total_loss.backward()
            tr_loss += total_loss.item()
            if device == "cpu":
                optimizer.step()
                scheduler.step()
            else:
                convert_models_to_fp32(model)
                optimizer.step()
                scheduler.step()
                clip.model.convert_weights(model)
            #wandb.log({'loss': total_loss.item()})
        tr_loss /= step

        if tr_loss < best_tr_loss:
            best_tr_loss = tr_loss
            best_ep = epoch


    #wandb.finish()
    torch.save(model.state_dict(), save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fine-tune CLIP model on dataset')
    parser.add_argument('--dataset', type=str, default='single', help='type of data: single, two-object or relational')
    parser.add_argument('--data_path', type=str, required=True, help='Path to dataset')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--save_path', type=str, help='Path to save the fine-tuned model, ending with .pt')
    args = parser.parse_args()

    finetune(dataset=args.dataset, seed=args.seed, data_path=args.data_path, save_path=args.save_path)