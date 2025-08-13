import torch
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

def evaluate_model(model, loader, vocab, caption_dict, max_len=20, num_samples=5, beam_size=3):
    model.eval()
    references, hypotheses = [], []
    sample_outputs = []
    with torch.no_grad():
        for batch_idx, (features, captions, image_names) in enumerate(tqdm(loader, desc="Evaluating")):
            if features is None:
                continue
            features, captions = features.cuda(), captions.cuda()
            outputs = model.inference(features, max_len=max_len, beam_size=beam_size)
            for i in range(captions.size(0)):
                image_name = image_names[i]
                ref = caption_dict.get(image_name,
                                       [vocab.itos[idx.item()] for idx in captions[i] if idx.item() not in [0, 1, 2]])
                hyp = [vocab.itos[idx.item()] for idx in outputs[i] if idx.item() not in [0, 1, 2]]
                references.append([ref])
                hypotheses.append(hyp)
                if len(sample_outputs) < num_samples and batch_idx * captions.size(0) + i < num_samples:
                    sample_outputs.append({
                        'image_name': image_name,
                        'ground_truth': ' '.join(ref),
                        'generated': ' '.join(hyp)
                    })
    smoothing = SmoothingFunction().method1
    bleu_score = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
    return bleu_score, sample_outputs