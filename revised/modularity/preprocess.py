import warnings
import os
import torch
import pandas as pd
import spacy
import pickle
from PIL import Image
import torchvision
import torchvision.models as models
from torchvision.models import ViT_B_16_Weights
from torch.utils.data import Subset, DataLoader
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
import time
from dataset import Flickr30kDataset, custom_collate_fn

warnings.filterwarnings('ignore')
torch.backends.cudnn.benchmark = True

# Image preprocessing pipeline
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_image(image_path):
    try:
        image = Image.open(image_path).convert('RGB')
        return image_transform(image)
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

def preprocess_caption(captions, batch_size=1000):
    captions = [str(c).strip() for c in captions if not pd.isna(c)]
    results = []
    for i in range(0, len(captions), batch_size):
        batch = captions[i:i + batch_size]
        docs = nlp.pipe([caption.lower() for caption in batch], disable=['parser', 'ner'])
        for doc in docs:
            tokens = [token.text for token in doc if not token.is_punct and token.text.strip()]
            results.append(tokens)
    return results

def extract_resnet_features(loader, split_name, subset_indices, feature_dir):
    resnet = models.resnet50(pretrained=True).cuda()
    resnet.eval()
    resnet = torch.nn.Sequential(*list(resnet.children())[:-1])  # Remove final layer
    os.makedirs(feature_dir, exist_ok=True)
    features = []
    captions_list = []
    image_names = []
    for batch_idx, (images, captions, batch_image_names) in enumerate(loader):
        if images is None:
            continue
        images = images.cuda()
        with torch.no_grad():
            batch_features = resnet(images).squeeze(-1).squeeze(-1)  # [batch_size, 2048]
        features.append(batch_features.cpu())
        captions_list.extend(captions)
        image_names.extend(batch_image_names)
    features = torch.cat(features, dim=0)
    torch.save({'features': features, 'captions': captions_list, 'image_names': image_names},
               os.path.join(feature_dir, f'{split_name}_features.pt'))
    print(f"Saved {split_name} ResNet features: {features.shape}")

def extract_vit_features(loader, split_name, feature_dir):
    vit = models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1).cuda()
    vit.heads = nn.Identity()  # Remove classification head
    vit.eval()
    os.makedirs(feature_dir, exist_ok=True)
    print(f"Starting {split_name} ViT feature extraction")
    features = []
    captions_list = []
    image_names = []
    start_time = time.time()
    with torch.amp.autocast('cuda'), torch.no_grad():
        for batch_idx, (images, captions, batch_image_names) in enumerate(
                tqdm(loader, desc=f"Processing {split_name}")):
            if images is None:
                print(f"Skipping batch {batch_idx + 1}: No images")
                continue
            images = images.cuda()
            batch_features = vit(images)  # [batch_size, 768]
            features.append(batch_features.cpu())
            captions_list.extend(captions)
            image_names.extend(batch_image_names)
        if features:
            final_features = torch.cat(features, dim=0)
        else:
            final_features = torch.tensor([]).reshape(0, 768)
        torchsave_path = os.path.join(feature_dir, f'{split_name}_vit_features.pt')
        torch.save({
            'features': final_features,
            'captions': captions_list,
            'image_names': image_names
        }, torchsave_path)
        print(f"Saved {split_name} ViT features: {final_features.shape}")
        print(f"Time taken: {(time.time() - start_time) / 60:.2f} minutes")

def preprocess_data(image_dir, annotation_file, resnet_feature_dir, vit_feature_dir):
    # Initialize dataset
    dataset = Flickr30kDataset(image_dir, annotation_file, image_transform)
    print("Dataset loaded, size:", len(dataset))

    # Create splits
    indices = np.arange(len(dataset))
    np.random.seed(42)
    np.random.shuffle(indices)
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    test_subset = Subset(dataset, test_indices)
    print("Train subset size:", len(train_subset))
    print("Val subset size:", len(val_subset))
    print("Test subset size:", len(test_subset))

    # DataLoaders
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_subset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)

    # Extract features
    extract_resnet_features(train_loader, 'train', train_indices, resnet_feature_dir)
    extract_resnet_features(val_loader, 'val', val_indices, resnet_feature_dir)
    extract_resnet_features(test_loader, 'test', test_indices, resnet_feature_dir)
    extract_vit_features(train_loader, 'train', vit_feature_dir)
    extract_vit_features(val_loader, 'val', vit_feature_dir)
    extract_vit_features(test_loader, 'test', vit_feature_dir)

    # Preprocess and save captions
    df = pd.read_csv(annotation_file, sep=',', names=['image', 'caption'], skiprows=1)
    preprocessed_captions = preprocess_caption(df['caption'].tolist())
    caption_file = os.path.join(vit_feature_dir, 'preprocessed_captions1.pkl')
    pickle.dump(preprocessed_captions, open(caption_file, 'wb'))
    print(f"Saved preprocessed captions to {caption_file}")

if __name__ == "__main__":
    image_dir = r'flickr30k/versions/1/Images'
    annotation_file = r'flickr30k/versions/1/captions.txt'
    resnet_feature_dir = r'flickr30k/versions/1/Features'
    vit_feature_dir = r'flickr30k/versions/1/ViT_Features'
    preprocess_data(image_dir, annotation_file, resnet_feature_dir, vit_feature_dir)