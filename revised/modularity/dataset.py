import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import pickle
from preprocess import preprocess_image, preprocess_caption, image_transform

class Flickr30kDataset(Dataset):
    def __init__(self, image_dir, annotation_file, image_transform=None):
        self.df = pd.read_csv(annotation_file, sep=',', names=['image', 'caption'], skiprows=1)
        self.df['caption'] = self.df['caption'].fillna('').astype(str)
        self.image_dir = image_dir
        self.image_transform = image_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['image']
        caption = row['caption']
        img_path = os.path.join(self.image_dir, img_name)
        image = preprocess_image(img_path)
        if image is None:
            return None
        caption_tokens = preprocess_caption([caption])[0]
        return image, caption_tokens, img_name

class Flickr30kCaptionDataset(Dataset):
    def __init__(self, feature_file, caption_file, augment=False):
        data = torch.load(feature_file, weights_only=False)
        self.features = data['features']
        self.image_names = data['image_names']
        self.captions_dict = pickle.load(open(caption_file, 'rb'))
        self.augment = augment
        from model import Vocabulary
        self.vocab = Vocabulary(min_freq=2)
        self.vocab.build_vocabulary(self.captions_dict)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        image_name = self.image_names[idx]
        captions = self.captions_dict.get(image_name, [[]])
        caption = captions[torch.randint(0, len(captions), (1,)).item()] if self.augment and len(captions) > 1 else captions[0]
        numerical_caption = [self.vocab.stoi['<SOS>']] + self.vocab.numericalize(caption) + [self.vocab.stoi['<EOS>']]
        return feature, torch.tensor(numerical_caption, dtype=torch.long), image_name

def custom_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None, None, None
    features, captions, image_names = zip(*batch)
    features = torch.stack(features)
    max_len = max(len(c) for c in captions)
    padded_captions = torch.zeros(len(captions), max_len, dtype=torch.long)
    for i, cap in enumerate(captions):
        padded_captions[i, :len(cap)] = cap
    return features, padded_captions, image_names