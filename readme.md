# Klasyfikacja klocków LEGO – Praca inżynierska

System do rozpoznawania klocków LEGO ze zdjęć.
Własny zbiór danych (ponad 6 tys. zdjęć, 20 klas) i proces uczenia od zera. Model to EfficientNet-B2 w PyTorch. Dzięki transfer learningowi i mocnej augmentacji danych osiąga 91.4% skuteczności na zbiorze testowym (zdjęcia w normalnych warunkach z tłem).

## Funkcje aplikacji

- **GUI:** Prosta aplikacja okienkowa w Tkinter i Matplotlib. Wybranie zdjęcia, otrzymanie Design ID klocka i pewność w %.
- **Detekcja nieznanych obiektów (OOD):** System odrzuca obiekty, których nie zna (np. monety). Liczy entropię Shannona z predykcji – jak wynik jest zbyt rozmyty, rzuca alert.
- **Kadrowanie na żywo:** Jeśli tło przeszkadza modelowi, opcja zaznaczenia kloceka myszką. Predykcja przelicza się w czasie rzeczywistym podczas rysowania ramki.
- **Test-Time Augmentation (TTA):** Podczas klasyfikacji aplikacja pod spodem obraca i odbija zdjęcie, a następnie uśrednia wyniki, żeby zniwelować nietypowe ułożenie klocka w kadrze.

## Skrypty i eksperymenty

- `train_lego.py` – główny skrypt uczący. Dwie fazy uczenia: 8 epok warmupu dla głowicy, potem 30 epok fine-tuningu całej sieci ze zmiennym LR (Cosine Annealing). Całość zoptymalizowana pod CUDA + AMP (mixed precision).
- `Predict.py` – apka predykcyjna z GUI. Wczytuje wagi, nakłada transformacje i odpala inferencję.
- `eksperyment1.py` – test wpływu augmentacji. Skrypt udowadnia, że dodanie rotacji i modyfikacji kolorów (Color Jitter) poprawia skuteczność o 6.7 pp i eliminuje overfitting.
- `ekperyment2.py` – benchmark GPU vs CPU. Potwierdza ponad 5-krotne przyspieszenie czasu uczenia pojedynczej epoki na CUDA (ze 722s na 137s).
- `eksperyment3.py` – porównanie architektur EfficientNet B0, B2 i B4. B2 wyszło jako najlepszy kompromis między precyzją predykcji a czasem uczenia.

## Stack

Python, PyTorch, Torchvision, NumPy, Matplotlib, Tkinter, PIL.

## Macierz pomyłek

![alt text](image.png)

## GUI

![alt text](image-1.png)
![alt text](image-2.png)
![alt text](image-4.png)
![alt text](image-3.png)
