import torch
import torch.nn as nn

class Vocabulary:
    def __init__(self, min_freq=2):
        self.itos = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>'}
        self.stoi = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
        self.min_freq = min_freq

    def build_vocabulary(self, captions):
        from collections import Counter
        words = [word for caption_list in captions.values() for caption in caption_list for word in caption]
        word_counts = Counter(words)
        idx = len(self.itos)
        for word, count in word_counts.items():
            if count >= self.min_freq and word not in self.stoi:
                self.itos[idx] = word
                self.stoi[word] = idx
                idx += 1

    def numericalize(self, caption):
        return [self.stoi.get(word, self.stoi['<UNK>']) for word in caption]

class ImageCaptionTransformer(nn.Module):
    def __init__(self, feature_dim, vocab_size, embed_dim=512, num_heads=8, num_encoder_layers=6, num_decoder_layers=6,
                 hidden_dim=2048, dropout=0.05, max_len=100):
        super(ImageCaptionTransformer, self).__init__()
        self.feature_fc = nn.Linear(feature_dim, embed_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoder = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim,
                                                   dropout=dropout, batch_first=True)
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim,
                                                   dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.fc = nn.Linear(embed_dim, vocab_size)
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.feature_fc.weight)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.feature_fc.bias)
        nn.init.zeros_(self.fc.bias)

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask

    def forward(self, features, captions=None, teacher_forcing_ratio=0.5):
        batch_size = features.size(0)
        device = features.device
        features = self.dropout(torch.relu(self.feature_fc(features)))
        features = features.unsqueeze(1)
        memory = self.transformer_encoder(features)
        if captions is None:
            return self.inference(features)
        captions = captions[:, :-1]
        embed = self.embedding(captions) * (self.embed_dim ** 0.5)
        positions = torch.arange(0, captions.size(1), device=device).unsqueeze(0).repeat(batch_size, 1)
        pos_embed = self.pos_encoder[:, :captions.size(1), :]
        embed = embed + pos_embed
        tgt_mask = self.generate_square_subsequent_mask(captions.size(1)).to(device)
        output = self.transformer_decoder(embed, memory, tgt_mask=tgt_mask)
        output = self.fc(output)
        return output

    def inference(self, features, max_len=20, beam_size=3):
        batch_size = features.size(0)
        device = features.device
        features = self.dropout(torch.relu(self.feature_fc(features)))
        features = features.unsqueeze(1)
        memory = self.transformer_encoder(features)
        sequences = [[torch.full((1, 1), 1, dtype=torch.long, device=device), 0.0]] * batch_size
        for _ in range(max_len):
            all_candidates = []
            for i in range(batch_size):
                candidates = []
                for seq, score in sequences[i]:
                    if seq[0, -1].item() == 2:
                        candidates.append([seq, score])
                        continue
                    embed = self.embedding(seq) * (self.embed_dim ** 0.5)
                    positions = torch.arange(0, seq.size(1), device=device).unsqueeze(0)
                    pos_embed = self.pos_encoder[:, :seq.size(1), :]
                    embed = embed + pos_embed
                    tgt_mask = self.generate_square_subsequent_mask(seq.size(1)).to(device)
                    output = self.transformer_decoder(embed, memory[i:i + 1], tgt_mask=tgt_mask)
                    output = self.fc(output[:, -1, :])
                    probs = torch.softmax(output, dim=-1)
                    top_probs, top_idx = probs.topk(beam_size, dim=-1)
                    for k in range(beam_size):
                        next_token = top_idx[0, k].unsqueeze(0).unsqueeze(0)
                        next_score = score - torch.log(top_probs[0, k]).item()
                        new_seq = torch.cat([seq, next_token], dim=1)
                        candidates.append([new_seq, next_score])
                candidates = sorted(candidates, key=lambda x: x[1])[:beam_size]
                all_candidates.append(candidates)
            sequences = all_candidates
        outputs = [sequences[i][0][0] for i in range(batch_size)]
        return torch.cat(outputs, dim=0)