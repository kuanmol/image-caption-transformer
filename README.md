# Flickr30k ViT Caption

An image captioning model using Vision Transformer (ViT) features and a Transformer-based architecture, trained on the Flickr30k dataset to generate descriptive captions for images.

<div align="center">
  <a href="https://github.com/your-username/flickr30k-vit-caption/stargazers">
    <img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/your-username/flickr30k-vit-caption">
  </a>
  <a href="https://github.com/your-username/flickr30k-vit-caption/blob/main/LICENSE">
    <img alt="License" src="https://img.shields.io/badge/license-MIT-blue">
  </a>
</div>

## Overview
This project implements an image captioning system that extracts features from images using a pre-trained Vision Transformer (ViT-B-16) and generates captions using a Transformer model with beam search. It is trained and evaluated on the Flickr30k dataset, achieving descriptive and contextually relevant captions.

### Features
- Extracts image features using ViT-B-16 and ResNet-50.
- Generates captions with a Transformer model using beam search.
- Evaluates performance with BLEU scores.
- Supports training with early stopping and cosine annealing learning rate scheduling.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/flickr30k-vit-caption.git
   cd flickr30k-vit-caption
   ```
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```
3. Download and extract the [Flickr30k dataset](https://www.kaggle.com/hsankesara/flickr-image-dataset) to `./flickr30k/versions/1/`.

## Usage
Run the pipeline using `src/main.py` with different modes:

1. **Preprocess**: Extract ResNet and ViT features and preprocess captions.
   ```bash
   python src/main.py --mode preprocess --image-dir flickr30k/versions/1/Images --caption-file flickr30k/versions/1/captions.txt --resnet-feature-dir flickr30k/versions/1/Features --vit-feature-dir flickr30k/versions/1/ViT_Features
   ```
2. **Train**: Train the Transformer model.
   ```bash
   python src/main.py --mode train --vit-feature-dir flickr30k/versions/1/ViT_Features --epochs 100
   ```
3. **Evaluate**: Evaluate the model and compute BLEU scores.
   ```bash
   python src/main.py --mode evaluate --vit-feature-dir flickr30k/versions/1/ViT_Features --caption-file flickr30k/versions/1/captions.txt
   ```

To generate a sample output image for the README:
1. Run the preprocessing step.
2. Use `utils.show_sample` with `save_path='sample_output.jpg'` to save an image and caption, then add `sample_output.jpg` to the repository.

## Dataset
The project uses the [Flickr30k dataset](https://www.kaggle.com/hsankesara/flickr-image-dataset), containing ~31,000 images with ~158,000 captions. Place the dataset in `./flickr30k/versions/1/Images` and `./flickr30k/versions/1/captions.txt`.

## Project Structure
```
flickr30k-vit-caption/
├── flickr30k/versions/1/
│   ├── Images/              # Flickr30k images
│   ├── Features/            # ResNet features
│   ├── ViT_Features/        # ViT features
│   ├── captions.txt         # Caption annotations
│   └── preprocessed_captions1.pkl  # Preprocessed captions
├── revised/
│   ├── preprocess.py        # Preprocessing and feature extraction
│   ├── model.py             # Transformer model and vocabulary
│   ├── dataset.py           # Dataset and dataloader definitions
│   ├── train.py             # Training and validation loops
│   ├── evaluate.py          # Evaluation with BLEU scores
│   ├── utils.py             # Utility functions
│   └── main.py              # Main script to run the pipeline
├── README.md                # Project documentation
├── LICENSE                  # License file
└── requirements.txt         # Dependencies
```

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-name`).
3. Commit your changes (`git commit -m 'Add feature'`).
4. Push to the branch (`git push origin feature-name`).
5. Open a pull request.


## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file in the root directory for details.

## Contributors
<a href="https://github.com/your-username/flickr30k-vit-caption/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=your-username/flickr30k-vit-caption" />
</a>

## Acknowledgments
- Flickr30k dataset for providing the training data.
- PyTorch and Hugging Face for model implementations.
- spaCy and NLTK for text preprocessing.

*Last updated: August 13, 2025*
