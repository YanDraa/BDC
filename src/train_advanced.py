import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import RandAugment
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import f1_score
import os

from dataset import WasteDataset
from model import get_model

# ================== CONFIG ==================
EXPERIMENT_NAME = "adv_efficientnet_b0_v1"
DATA_DIR = 'data/train'
TEST_DIR = 'data/test'
BATCH_SIZE = 16
NUM_EPOCHS = 15
LEARNING_RATE = 1e-3
MODEL_NAME = 'efficientnet_b0'

# --- Anti-putus: semua checkpoint & log disimpan ke Google Drive ---
# GANTI path ini sesuai folder Drive kamu. Pastikan drive.mount() sudah
# dijalankan di cell sebelumnya sebelum script ini dipanggil.
DRIVE_SAVE_DIR = '/content/drive/MyDrive/BDC/models'
LAST_CKPT_PATH = os.path.join(DRIVE_SAVE_DIR, f'last_{EXPERIMENT_NAME}.pth')
BEST_CKPT_PATH = os.path.join(DRIVE_SAVE_DIR, f'best_{EXPERIMENT_NAME}.pth')
HISTORY_CSV_PATH = os.path.join(DRIVE_SAVE_DIR, f'history_{EXPERIMENT_NAME}.csv')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE} | Experiment: {EXPERIMENT_NAME}")

# Pastikan folder models selalu ada -- baik folder lokal Colab maupun di Drive.
# Ini mencegah error "Parent directory models does not exist" kalau runtime
# sempat disconnect/reconnect atau cell dijalankan tidak berurutan.
os.makedirs('models', exist_ok=True)
os.makedirs(DRIVE_SAVE_DIR, exist_ok=True)

# Strong Augmentation
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    RandAugment(num_ops=2, magnitude=9),   # Strong augmentation
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Dataset
full_dataset = WasteDataset(DATA_DIR, transform=None)
train_idx, val_idx = train_test_split(range(len(full_dataset)), test_size=0.15, stratify=full_dataset.labels, random_state=42)

train_dataset = WasteDataset(DATA_DIR, transform=train_transform)
train_dataset.images = [full_dataset.images[i] for i in train_idx]
train_dataset.labels = [full_dataset.labels[i] for i in train_idx]

val_dataset = WasteDataset(DATA_DIR, transform=val_transform)
val_dataset.images = [full_dataset.images[i] for i in val_idx]
val_dataset.labels = [full_dataset.labels[i] for i in val_idx]

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

# Model
model = get_model(MODEL_NAME, 3).to(DEVICE)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

best_f1 = 0.0
start_epoch = 0
history = []

# --- Resume otomatis kalau ada checkpoint tersisa dari sesi sebelumnya ---
if os.path.exists(LAST_CKPT_PATH):
    print(f"Checkpoint ditemukan di {LAST_CKPT_PATH}, melanjutkan training...")
    checkpoint = torch.load(LAST_CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_f1 = checkpoint['best_f1']
    print(f"Melanjutkan dari epoch {start_epoch + 1}, best_f1 sebelumnya: {best_f1:.5f}")
else:
    print("Tidak ada checkpoint sebelumnya, training dari awal.")

if os.path.exists(HISTORY_CSV_PATH):
    history = pd.read_csv(HISTORY_CSV_PATH).to_dict('records')

print("Training Advanced dimulai...")

for epoch in range(start_epoch, NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    # Validation
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    acc = (sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)) * 100
    
    print(f'Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {running_loss/len(train_loader):.4f} | Val Acc: {acc:.2f}% | Macro F1: {macro_f1:.5f}')

    is_best = macro_f1 > best_f1
    if is_best:
        best_f1 = macro_f1
        print(f"Best model baru! Macro F1: {best_f1:.5f}")

    # --- Checkpoint lengkap (model + optimizer + scheduler + epoch) ---
    # Disimpan SETIAP epoch ke Google Drive, supaya kalau runtime putus
    # di tengah jalan, training bisa lanjut dari epoch terakhir yang selesai.
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_f1': best_f1,
    }
    torch.save(checkpoint, LAST_CKPT_PATH)

    if is_best:
        torch.save(model.state_dict(), BEST_CKPT_PATH)
        # Simpan juga salinan lokal (opsional, cepat diakses di sesi ini)
        torch.save(model.state_dict(), f'models/best_{EXPERIMENT_NAME}.pth')

    # --- Log history training ke CSV, supaya bisa dipantau/diplot nanti ---
    history.append({
        'epoch': epoch + 1,
        'train_loss': running_loss / len(train_loader),
        'val_acc': acc,
        'val_macro_f1': macro_f1,
    })
    pd.DataFrame(history).to_csv(HISTORY_CSV_PATH, index=False)

    scheduler.step()

print(f"Training selesai! Best Macro F1: {best_f1:.5f}")
print(f"Checkpoint terakhir: {LAST_CKPT_PATH}")
print(f"Model terbaik: {BEST_CKPT_PATH}")
print(f"History training: {HISTORY_CSV_PATH}")