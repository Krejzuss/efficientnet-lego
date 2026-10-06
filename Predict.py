import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageOps
import os
import tkinter as tk
from tkinter import filedialog, messagebox
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import sys



# 1. KONFIGURACJA

script_dir  = os.path.dirname(os.path.abspath(__file__))
model_path  = os.path.join(script_dir, 'lego_efficientnet_b2_best.pth')
train_dir   = os.path.abspath(os.path.join(script_dir, '..', 'dataset_lego', 'train'))
img_size    = 260
THRESHOLD   = 75.0
ENTROPY_MAX = 2.0   

plt.rcParams['toolbar'] = 'None'


# 2. ODCZYTANIE NAZW KLAS

try:
    class_names = sorted([d for d in os.listdir(train_dir)
                          if os.path.isdir(os.path.join(train_dir, d))])
    num_classes = len(class_names)
    print(f"Wczytano {num_classes} klas.")
except FileNotFoundError:
    print(f"BŁĄD: Nie znaleziono folderu: {train_dir}")
    exit()


# 3. MODEL: EfficientNet-B2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Urządzenie: {device}")

model = models.efficientnet_b2()
num_ftrs = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.3, inplace=True),
    nn.Linear(num_ftrs, num_classes)
)

try:
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    print("Wagi modelu wczytane pomyślnie.")
except FileNotFoundError:
    print(f"BŁĄD: Nie znaleziono pliku modelu: {model_path}")
    exit()

model = model.to(device)
model.eval()


# 4. TRANSFORMACJE

image_transforms = transforms.Compose([
    transforms.Resize(int(img_size * 1.1)),
    transforms.CenterCrop(img_size),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# 6. RĘCZNE KADROWANIE Z PRZYCISKAMI

def manual_crop_with_preview(pil_img: Image.Image):
    from matplotlib.widgets import RectangleSelector, Button
    import threading

    result = {'crop': None, 'top5': None}
    prediction_lock = threading.Lock()

    
    fig = plt.figure(figsize=(6, 8))
    
    
    fig.canvas.manager.set_window_title('ręczne kadrowanie')
    
    fig.suptitle("Zaznacz klocek na zdjęciu",
                 fontsize=11, color='navy', fontweight='bold')

    
    ax_img     = fig.add_axes([0.0, 0.12, 1.0, 0.76])
    ax_confirm = fig.add_axes([0.15, 0.02, 0.40, 0.06])
    ax_cancel  = fig.add_axes([0.60, 0.02, 0.25, 0.06])

    ax_img.imshow(pil_img)
    ax_img.axis('off')
    

    btn_confirm = Button(ax_confirm, 'Zatwierdź zaznaczenie', color='#2ecc71', hovercolor='#27ae60')
    btn_cancel  = Button(ax_cancel,  'Pomiń', color='#e74c3c', hovercolor='#c0392b')

    btn_confirm.label.set_fontsize(10)
    btn_cancel.label.set_fontsize(10)

    def run_prediction(crop_img):
        tensor = image_transforms(crop_img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(tensor)
            probs   = F.softmax(outputs, dim=1)
        top5_probs, top5_idx = torch.topk(probs, min(5, num_classes), dim=1)
        return [(class_names[i], p * 100) for i, p in
                zip(top5_idx[0].tolist(), top5_probs[0].tolist())]

    def on_select(eclick, erelease):
        x1 = int(min(eclick.xdata, erelease.xdata))
        y1 = int(min(eclick.ydata, erelease.ydata))
        x2 = int(max(eclick.xdata, erelease.xdata))
        y2 = int(max(eclick.ydata, erelease.ydata))
        if x2 - x1 < 10 or y2 - y1 < 10:
            return

        def predict_and_update():
            with prediction_lock:
                img_to_predict = pil_img.crop((x1, y1, x2, y2))
                top5 = run_prediction(img_to_predict)
                result['crop'] = img_to_predict
                result['top5'] = top5
            best_class, best_conf = top5[0]
            color = 'green' if best_conf >= THRESHOLD else 'red'
            ax_img.set_title(f"{best_class}  |  {best_conf:.1f}%",
                             fontsize=16, fontweight='bold', color=color)
            fig.canvas.draw_idle()

        t = threading.Thread(target=predict_and_update, daemon=True)
        t.start()

    def on_confirm(event):
        if result['crop'] is not None:
            plt.close(fig)

    def on_cancel(event):
        result['crop'] = None
        result['top5'] = None
        plt.close(fig)

    def on_key(event):
        if event.key == 'escape':
            on_cancel(None)
        elif event.key == 'enter':
            on_confirm(None)

    btn_confirm.on_clicked(on_confirm)
    btn_cancel.on_clicked(on_cancel)

    selector = RectangleSelector(
        ax_img, on_select,
        useblit=True, button=[1],
        minspanx=10, minspany=10,
        spancoords='pixels', interactive=True,
        props=dict(facecolor='yellow', edgecolor='red', alpha=0.3, fill=True)
    )

    fig.canvas.mpl_connect('key_press_event', on_key)
    
    plt.show()

    return result['crop'], result['top5']


# 7. GŁÓWNA PREDYKCJA I INTEGRACJA LOGIKI (z TTA)

def predict_image(image_path, manual_crop_img=None):
    img = Image.open(image_path).convert('RGB')
    img = ImageOps.exif_transpose(img)
    img_display = img.copy()
    competition_conf = None

    if manual_crop_img is not None:
        base_img = manual_crop_img
    else:
        base_img = img

    
    tta_images = [
        base_img,
        ImageOps.mirror(base_img),
        base_img.rotate(12, expand=False),
        base_img.rotate(-12, expand=False),
    ]
    tensor_batch = torch.stack([image_transforms(i) for i in tta_images]).to(device)

    with torch.no_grad():
        outputs   = model(tensor_batch)
        probs     = F.softmax(outputs, dim=1)
        avg_probs = torch.mean(probs, dim=0, keepdim=True)

    top5_probs, top5_idx = torch.topk(avg_probs, min(5, num_classes), dim=1)
    top5 = [(class_names[i], p * 100) for i, p in
            zip(top5_idx[0].tolist(), top5_probs[0].tolist())]

    
    probs_np = avg_probs[0].cpu().numpy() + 1e-9
    entropy  = float(-np.sum(probs_np * np.log(probs_np)))
    print(f"  Entropia: {entropy:.3f} (próg: {ENTROPY_MAX})")

    
    if competition_conf is not None and competition_conf > top5[0][1]:
        print(f"  TTA: {top5[0][1]:.1f}% → używam lepszego wyniku z zawodów: {competition_conf:.1f}%")
        top5[0] = (top5[0][0], competition_conf)

    return top5, img_display, base_img, entropy


# 8. GUI

def select_and_predict():
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    file_path = filedialog.askopenfilename(
        title="Wybierz zdjęcie klocka LEGO",
        filetypes=[("Obrazy", "*.jpg *.jpeg *.png"), ("Wszystkie", "*.*")]
    )

    if not file_path:
        print("Nie wybrano pliku.")
        root.destroy()
        return

    print(f"\nAnalizuję: {file_path}")

    top5, img_display, img_cropped, entropy = predict_image(file_path)
    best_class, best_conf = top5[0]

  
    unknown = entropy > ENTROPY_MAX or best_conf < THRESHOLD

    
    if unknown:
        reason = f"entropia: {entropy:.2f} > {ENTROPY_MAX}" if entropy > ENTROPY_MAX else f"pewność: {best_conf:.1f}% < {THRESHOLD}%"
        answer = messagebox.askyesno(
            "Nieznany klocek?",
            f"Model nie rozpoznał klocka pewnie ({reason}).\n\n"
            "Możliwe że tego klocka nie ma w bazie.\n\n"
            "Czy chcesz spróbować ręcznie zaznaczyć klocek?",
            parent=root
        )
        if answer:
            img_orig = Image.open(file_path).convert('RGB')
            img_orig = ImageOps.exif_transpose(img_orig)
            manual_crop_img, manual_top5 = manual_crop_with_preview(img_orig)
            if manual_crop_img is not None and manual_top5 is not None:
                top5        = manual_top5
                img_cropped = manual_crop_img
                best_class, best_conf = top5[0]
                tensor = image_transforms(manual_crop_img).unsqueeze(0).to(device)
                with torch.no_grad():
                    probs_manual = F.softmax(model(tensor), dim=1)[0].cpu().numpy()
                probs_manual += 1e-9
                entropy = float(-np.sum(probs_manual * np.log(probs_manual)))
                unknown = entropy > ENTROPY_MAX or best_conf < THRESHOLD

    above_threshold = best_conf >= THRESHOLD and not unknown

    print("=" * 50)
    if unknown:
        print(f"  NIEZNANY KLOCEK | Entropia: {entropy:.3f} | Pewność: {best_conf:.1f}%")
        print(f"  Najbliższa klasa: {best_class}")
    else:
        print(f"  Klasa: {best_class}  |  Pewność: {best_conf:.1f}%  |  Entropia: {entropy:.3f}")
    print("=" * 50)

    if unknown:
        title    = "Nieznany klocek\n"
        subtitle = f"Brak w bazie  |  Entropia: {entropy:.2f}\nNajbliższa klasa: {best_class} ({best_conf:.1f}%)"
        color    = '#e67e22'
    elif above_threshold:
        title    = f"{best_class}"
        subtitle = f"{best_conf:.1f}%"
        color    = '#2ecc71'
    else:
        title    = f"{best_class}?"
        subtitle = f"{best_conf:.1f}%  (poniżej progu {THRESHOLD}%)"
        color    = '#e74c3c'

    fig, axes = plt.subplots(
    2, 1,
    figsize=(6, 8),
    gridspec_kw={'height_ratios': [4, 1]}  
)
    

    fig.canvas.manager.set_window_title("Klasyfikacja klocków LEGO")

    axes[0].imshow(img_display)
    axes[0].axis('off')
    axes[0].set_title('Zdjęcie', fontsize=10, color='white')

    axes[1].set_facecolor('#1a1a1a')
    axes[1].axis('off')
    axes[1].text(0.5, 0.6, title,
                 ha='center', va='center', fontsize=28, fontweight='bold',
                 color=color, transform=axes[1].transAxes)
    axes[1].text(0.5, 0.35, subtitle,
                 ha='center', va='center', fontsize=20,
                 color=color, transform=axes[1].transAxes)

    plt.tight_layout()
    plt.show()
    root.destroy()

if __name__ == '__main__':
    select_and_predict()