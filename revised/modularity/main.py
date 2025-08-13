import argparse
from preprocess import preprocess_data
from dataset import Flickr30kCaptionDataset, custom_collate_fn
from model import ImageCaptionTransformer
from train import train_model
from evalutate import evaluate_model
from utils import load_original_captions
from torch.utils.data import DataLoader
import torch
import pickle
import os

def main(args):
    feature_dir = args.vit_feature_dir
    caption_file = os.path.join(feature_dir, 'preprocessed_captions1.pkl')
    caption_txt_file = args.caption_file

    if args.mode == 'preprocess':
        preprocess_data(args.image_dir, args.caption_file, args.resnet_feature_dir, args.vit_feature_dir)
    elif args.mode == 'train':
        train_dataset = Flickr30kCaptionDataset(os.path.join(feature_dir, 'train_vit_features.pt'), caption_file, augment=True)
        val_dataset = Flickr30kCaptionDataset(os.path.join(feature_dir, 'val_vit_features.pt'), caption_file, augment=False)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, collate_fn=custom_collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)
        model = ImageCaptionTransformer(
            feature_dim=768,
            vocab_size=len(train_dataset.vocab.itos),
            embed_dim=512,
            num_heads=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            hidden_dim=2048,
            dropout=0.05,
            max_len=100
        ).cuda()
        train_model(model, train_loader, val_loader, feature_dir, num_epochs=args.epochs)
        with open(os.path.join(feature_dir, 'vocab.pkl'), 'wb') as f:
            pickle.dump(train_dataset.vocab, f)
        print(f"Saved vocab.pkl with {len(train_dataset.vocab.itos)} tokens")
    elif args.mode == 'evaluate':
        test_dataset = Flickr30kCaptionDataset(os.path.join(feature_dir, 'test_vit_features.pt'), caption_file, augment=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)
        model = ImageCaptionTransformer(
            feature_dim=768,
            vocab_size=len(test_dataset.vocab.itos),
            embed_dim=512,
            num_heads=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            hidden_dim=2048,
            dropout=0.05,
            max_len=100
        ).cuda()
        model.load_state_dict(torch.load(os.path.join(feature_dir, 'best_transformer_model.pt')))
        caption_dict = load_original_captions(caption_txt_file)
        bleu_score, sample_outputs = evaluate_model(model, test_loader, test_dataset.vocab, caption_dict)
        print(f"BLEU Score: {bleu_score:.4f}")
        for output in sample_outputs:
            print(f"Image: {output['image_name']}")
            print(f"Ground Truth: {output['ground_truth']}")
            print(f"Generated: {output['generated']}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Captioning with Flickr30k and ViT")
    parser.add_argument('--mode', choices=['preprocess', 'train', 'evaluate'], required=True, help="Mode to run: preprocess, train, or evaluate")
    parser.add_argument('--image-dir', default='flickr30k/versions/1/Images', help="Path to Flickr30k images")
    parser.add_argument('--caption-file', default='flickr30k/versions/1/captions.txt', help="Path to caption file")
    parser.add_argument('--resnet-feature-dir', default='flickr30k/versions/1/Features', help="Path to save ResNet features")
    parser.add_argument('--vit-feature-dir', default='flickr30k/versions/1/ViT_Features', help="Path to save ViT features")
    parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs")
    args = parser.parse_args()
    main(args)