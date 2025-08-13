import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model import ImageCaptionTransformer

def train_epoch(model, loader, criterion, optimizer, teacher_forcing_ratio=0.5):
    model.train()
    total_loss = 0
    for features, captions, _ in tqdm(loader, desc="Training"):
        if features is None:
            continue
        features, captions = features.cuda(), captions.cuda()
        outputs = model(features, captions, teacher_forcing_ratio)
        loss = criterion(outputs.view(-1, outputs.size(-1)), captions[:, 1:].contiguous().view(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def validate_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for features, captions, _ in tqdm(loader, desc="Validating"):
            if features is None:
                continue
            features, captions = features.cuda(), captions.cuda()
            outputs = model(features, captions, teacher_forcing_ratio=1.0)
            loss = criterion(outputs.view(-1, outputs.size(-1)), captions[:, 1:].contiguous().view(-1))
            total_loss += loss.item()
    return total_loss / len(loader)

def train_model(model, train_loader, val_loader, feature_dir, num_epochs=100):
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)
    best_val_loss = float('inf')
    patience = 3
    no_improve_count = 0
    min_delta = 0.01
    import time
    for epoch in range(num_epochs):
        start_time = time.time()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, teacher_forcing_ratio=0.5)
        val_loss = validate_epoch(model, val_loader, criterion)
        scheduler.step()
        print(
            f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
            f"Time: {(time.time() - start_time) / 60:.2f} min, LR: {optimizer.param_groups[0]['lr']:.6f}")
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            no_improve_count = 0
            torch.save(model.state_dict(), os.path.join(feature_dir, 'best_transformer_model.pt'))
            print(f"Saved best model with Val Loss: {best_val_loss:.4f}")
        else:
            no_improve_count += 1
            print(f"No improvement in Val Loss, count: {no_improve_count}/{patience}")
        if no_improve_count >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break