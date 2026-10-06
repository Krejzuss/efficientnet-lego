import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os
import time
import sys
import json
import argparse
import csv

if __name__ == '__main__':
    
    # 1. WYBÓR URZĄDZENIA I ARGUMENTY
    

    parser = argparse.ArgumentParser(description='Trening modelu LEGO')
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    args = parser.parse_args()

    if args.device == 'auto':
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif args.device == 'cuda':
        if not torch.cuda.is_available():
            sys.exit("BŁĄD: Wybrano cuda, ale karta graficzna nie jest dostępna.")
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    use_amp = device.type == 'cuda'

    print(f"Urządzenie: {device}")
    if not use_amp:
        print("AMP (mixed precision): wyłączone")

    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')

    # Seed dla reprodukowalności
    SEED = 42
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    import random; random.seed(SEED)

   
    # 2. KONFIGURACJA I ŚCIEŻKI
    

    script_dir = os.path.dirname(os.path.abspath(__file__))

    
    data_dir_train = os.path.abspath(os.path.join(script_dir, '..', 'dataset_lego'))
    data_dir_val   = os.path.abspath(os.path.join(script_dir, '..', 'dataset_lego'))

    img_size   = 260
    batch_size = 64
    num_epochs = 30
    warmup_epochs = 8
    lr_head     = 0.001
    lr_finetune = 0.00005
    early_stopping_patience = 12

    workers = 6

    config = {
        'model': 'EfficientNet-B2',
        'device': str(device),
        'use_amp': use_amp,
        'img_size': img_size,
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'lr_head': lr_head,
        'lr_finetune': lr_finetune,
        'early_stopping_patience': early_stopping_patience,
        'mixup': False,
        'label_smoothing': False,
        'augmentacje': 'uproszczone',
    }
    with open(os.path.join(script_dir, 'training_config.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    
    # 3. TRANSFORMACJE I DANE
    

    data_transforms = {
        
        'train': transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=12, scale=(0.9, 1.1), translate=(0.05, 0.05)),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        
        'val': transforms.Compose([
            transforms.Resize(int(img_size * 1.1)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {
        'train': datasets.ImageFolder(os.path.join(data_dir_train, 'train'), data_transforms['train']),
        'val':   datasets.ImageFolder(os.path.join(data_dir_val,   'val'),   data_transforms['val']),
    }

    if image_datasets['train'].classes != image_datasets['val'].classes:
        sys.exit(f"BŁĄD: Różne klasy w train i val!\n"
                 f"Train: {image_datasets['train'].classes}\n"
                 f"Val:   {image_datasets['val'].classes}")

    dataloaders = {
        'train': DataLoader(
            image_datasets['train'],
            batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=True,
            persistent_workers=True, prefetch_factor=2,
        ),
        'val': DataLoader(
            image_datasets['val'],
            batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=True,
            persistent_workers=True, prefetch_factor=2,
        ),
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names   = image_datasets['train'].classes
    num_classes   = len(class_names)

    print(f"\nModel: EfficientNet-B2 | Urządzenie: {device} | AMP: {use_amp}")
    print(f"Zdjęć: {dataset_sizes['train']} (train), {dataset_sizes['val']} (val)")
    print(f"Klas: {num_classes} — {class_names}")
    print(f"Batch: {batch_size} | MixUp: NIE | Label smoothing: NIE | Augmentacje: uproszczone\n")

    
    # 4. TTA — Test Time Augmentation dla val
    

    def evaluate_with_tta(model, loader):
        
        import torchvision.transforms.functional as TF

        model.eval()
        all_probs  = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(device, non_blocking=True)
                all_labels.append(labels)

                variants = [
                    inputs,
                    TF.rotate(inputs, angle=10),
                    TF.rotate(inputs, angle=-10),
                ]

                batch_probs = None
                for v in variants:
                    with autocast('cuda', enabled=use_amp):
                        out = model(v)
                    p = torch.softmax(out, dim=1)
                    batch_probs = p if batch_probs is None else batch_probs + p

                all_probs.append((batch_probs / len(variants)).cpu())

        all_probs  = torch.cat(all_probs,  dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        preds      = all_probs.argmax(dim=1)
        return (preds == all_labels).float().mean().item()

    
    # 5. MODEL I FUNKCJA STRATY
    

    model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)

    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(num_ftrs, num_classes)
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()  
    scaler    = torch.amp.GradScaler('cuda', enabled=use_amp)
    history   = []

    
    # 6. FAZA 1: WARMUP     

    print(f"{'='*55}")
    print(f"FAZA 1: Warmup — tylko głowa ({warmup_epochs} epok)")
    print(f"{'='*55}\n")

    optimizer = optim.AdamW(model.classifier.parameters(), lr=lr_head, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=warmup_epochs)
    best_acc  = 0.0

    for epoch in range(warmup_epochs):
        epoch_start = time.time()
        current_lr  = optimizer.param_groups[0]['lr']
        print(f'Epoka {epoch+1}/{warmup_epochs}  |  LR: {current_lr:.6f}')

        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss, running_corrects = 0.0, 0
            optimizer.zero_grad(set_to_none=True)

            pbar = tqdm(dataloaders[phase], desc=f'  {phase:5s}', leave=False,
                        unit='batch', colour='green' if phase == 'train' else 'cyan')

            for step, (inputs, labels) in enumerate(pbar):
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with torch.set_grad_enabled(phase == 'train'):
                    with autocast('cuda', enabled=use_amp):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                    if phase == 'train':
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)

                running_loss     += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc  = (running_corrects.double() / dataset_sizes[phase]).item()
            print(f'  {phase:5s} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), os.path.join(script_dir, 'lego_efficientnet_b2_best.pth'))
                print(f'  ✓ Nowy rekord warmup: {best_acc:.4f} — model zapisany')
        print(f'  ⏱  Czas epoki: {time.time() - epoch_start:.1f}s\n')

    print(f'Najlepsza acc po warmupie: {best_acc:.4f}\n')

    
    # 7. FAZA 2: FINE-TUNING (cały model)
    

    print(f"{'='*55}")
    print(f"FAZA 2: Fine-tuning całej sieci (max {num_epochs} epok)")
    print(f"{'='*55}\n")

    for param in model.parameters():
        param.requires_grad = True

    optimizer = optim.AdamW([
        {'params': model.features.parameters(),   'lr': lr_finetune},       
        {'params': model.classifier.parameters(), 'lr': lr_finetune * 10},  
    ], weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-7)

    epochs_no_improve = 0
    since = time.time()

    for epoch in range(num_epochs):
        epoch_start = time.time()
        lr_backbone = optimizer.param_groups[0]['lr']
        lr_head_ft  = optimizer.param_groups[1]['lr']
        print(f'Epoka {epoch+1}/{num_epochs}  |  LR backbone: {lr_backbone:.7f}  |  LR głowa: {lr_head_ft:.6f}')

        phase_losses, phase_accs = {}, {}
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss, running_corrects = 0.0, 0
            optimizer.zero_grad(set_to_none=True)

            pbar = tqdm(dataloaders[phase], desc=f'  {phase:5s}', leave=False,
                        unit='batch', colour='green' if phase == 'train' else 'cyan')

            for step, (inputs, labels) in enumerate(pbar):
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with torch.set_grad_enabled(phase == 'train'):
                    with autocast('cuda', enabled=use_amp):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                    if phase == 'train':
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)

                running_loss     += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc  = (running_corrects.double() / dataset_sizes[phase]).item()
            phase_losses[phase] = round(epoch_loss, 4)
            phase_accs[phase]   = round(epoch_acc,  4)
            print(f'  {phase:5s} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}')

        
        if (epoch + 1) % 5 == 0 or epochs_no_improve == 0 or epoch == num_epochs - 1:
            tta_acc = evaluate_with_tta(model, dataloaders['val'])
            print(f'  val TTA | Acc: {tta_acc:.4f}')
        else:
            tta_acc = phase_accs['val']

        if tta_acc > best_acc:
            best_acc = tta_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(script_dir, 'lego_efficientnet_b2_best.pth'))
            print(f'  ✓ Nowy rekord! {best_acc:.4f} — model zapisany')
        else:
            epochs_no_improve += 1
            print(f'  Brak poprawy: {epochs_no_improve}/{early_stopping_patience}')

        
        torch.save({
            'epoch':       epoch + 1,
            'model_state': model.state_dict(),
            'optimizer':   optimizer.state_dict(),
            'scheduler':   scheduler.state_dict(),
            'best_acc':    best_acc,
        }, os.path.join(script_dir, 'lego_efficientnet_b2_last.pth'))

        scheduler.step()

        epoch_time    = time.time() - epoch_start
        total_elapsed = time.time() - since
        eta           = epoch_time * (num_epochs - (epoch + 1))
        print(f'  ⏱  Czas epoki: {epoch_time:.1f}s  |  Łącznie: {total_elapsed/60:.1f}min  |  ETA: {eta/60:.1f}min\n')

        history.append({
            'epoch':       epoch + 1,
            'train_loss':  phase_losses['train'],
            'train_acc':   phase_accs['train'],
            'val_loss':    phase_losses['val'],
            'val_acc':     phase_accs['val'],
            'val_tta_acc': round(tta_acc, 4),
            'epoch_time':  round(epoch_time, 1),
            'lr_backbone': round(lr_backbone, 7),
        })

        if epochs_no_improve >= early_stopping_patience:
            print(f'Early stopping po {epoch+1} epokach (brak poprawy przez {early_stopping_patience} epok)')
            break

    time_elapsed = time.time() - since
    print(f'\nTrening zakończony w {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s')
    print(f'Najlepsza dokładność val (TTA): {best_acc:.4f}')

   
    # 8. ZAPIS WYNIKÓW DO CSV

    csv_path = os.path.join(script_dir, 'training_history.csv')
    csv_fields = ['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'val_tta_acc', 'epoch_time', 'lr_backbone']

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(history)

    print(f"Historia zapisana: {csv_path}")

   
    # 9. CONFUSION MATRIX NA VAL
    

    print("\nGenerowanie confusion matrix...")
    import matplotlib.pyplot as plt

    model.load_state_dict(torch.load(
        os.path.join(script_dir, 'lego_efficientnet_b2_best.pth'),
        map_location=device, weights_only=True
    ))
    model.eval()

    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloaders['val']:
            inputs = inputs.to(device, non_blocking=True)
            with autocast('cuda', enabled=use_amp):
                outputs = model(inputs)
            preds = outputs.argmax(dim=1).cpu()
            all_preds.append(preds)
            all_labels.append(labels)

    all_preds  = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    n = num_classes
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(all_labels, all_preds):
        cm[t][p] += 1

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm  = np.divide(cm.astype(float), row_sums, where=row_sums > 0)

    fig, axes = plt.subplots(1, 2, figsize=(max(12, n), max(6, n // 2 + 4)))
    fig.suptitle(f'Confusion Matrix — val  |  Najlepsza acc TTA: {best_acc:.4f}',
                 fontsize=13, fontweight='bold')

    im1 = axes[0].imshow(cm, cmap='Blues')
    axes[0].set_title('Liczby bezwzględne')
    axes[0].set_xticks(range(n)); axes[0].set_yticks(range(n))
    axes[0].set_xticklabels(class_names, rotation=45, ha='right', fontsize=8)
    axes[0].set_yticklabels(class_names, fontsize=8)
    axes[0].set_ylabel('Prawdziwa klasa')
    axes[0].set_xlabel('Przewidziana klasa')
    for i in range(n):
        for j in range(n):
            v = cm[i, j]
            if v > 0:
                axes[0].text(j, i, str(v), ha='center', va='center', fontsize=7,
                             color='white' if v > cm.max() * 0.6 else 'black',
                             fontweight='bold' if i == j else 'normal')
    plt.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(cm_norm, cmap='RdYlGn', vmin=0, vmax=1)
    axes[1].set_title('Recall per klasa (%)')
    axes[1].set_xticks(range(n)); axes[1].set_yticks(range(n))
    axes[1].set_xticklabels(class_names, rotation=45, ha='right', fontsize=8)
    axes[1].set_yticklabels(class_names, fontsize=8)
    axes[1].set_ylabel('Prawdziwa klasa')
    axes[1].set_xlabel('Przewidziana klasa')
    for i in range(n):
        for j in range(n):
            v = cm_norm[i, j]
            if v > 0:
                axes[1].text(j, i, f'{v:.0%}', ha='center', va='center', fontsize=7,
                             color='white' if v < 0.4 or v > 0.75 else 'black',
                             fontweight='bold' if i == j else 'normal')
    plt.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    cm_path = os.path.join(script_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix zapisana: {cm_path}")