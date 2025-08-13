import torch
import numpy as np
import matplotlib.pyplot as plt

def show_sample(image, caption):
    try:
        if not isinstance(image, torch.Tensor):
            raise ValueError("Image is not a torch.Tensor")
        if image.shape != (3, 224, 224):
            raise ValueError(f"Unexpected image shape: {image.shape}")
        image = image.permute(1, 2, 0).numpy()
        image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        image = np.clip(image, 0, 1)
        if not isinstance(caption, list) or not all(isinstance(word, str) for word in caption):
            raise ValueError(f"Invalid caption format: {caption}")
        caption_text = ' '.join(caption).strip()
        plt.imshow(image)
        plt.title(caption_text)
        plt.axis('off')
        plt.show()
    except Exception as e:
        print(f"Error in show_sample: {e}")

def load_original_captions(caption_txt_file):
    caption_dict = {}
    try:
        with open(caption_txt_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()[1:]  # Skip header
            for line in lines:
                parts = line.strip().split(',', 1)
                if len(parts) == 2:
                    image_name, caption = parts
                    caption_dict[image_name] = caption.split()
    except UnicodeDecodeError:
        with open(caption_txt_file, 'r', encoding='latin1') as f:
            lines = f.readlines()[1:]  # Skip header
            for line in lines:
                parts = line.strip().split(',', 1)
                if len(parts) == 2:
                    image_name, caption = parts
                    caption_dict[image_name] = caption.split()
    return caption_dict