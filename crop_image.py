import cv2
import numpy as np
import os
import uuid

# --- FUNKCJE POMOCNICZE (LOGIKA OBRÓBKI) ---

def _apply_hardcore_processing(img_array):
    """Wewnętrzna funkcja: CLAHE + Wyostrzanie"""
    # 1. CLAHE (Kontrast)
    clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(8, 8))
    final = clahe.apply(img_array)

    # 2. Wyostrzanie
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    final_sharpened = cv2.filter2D(final, -1, kernel, borderType=cv2.BORDER_REPLICATE)
    return final_sharpened

def _get_coordinates(img):
    """Wewnętrzna funkcja: Znajdowanie bounding boxa"""
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    # Tło=255, Obiekt=0 -> THRESH_BINARY_INV
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    
    margin = 20
    h_img, w_img = img.shape
    x_start = max(x - margin, 0)
    y_start = max(y - margin, 0)
    x_end = min(x + w + margin, w_img)
    y_end = min(y + h + margin, h_img)
    
    return (x_start, y_start, x_end, y_end)

# --- GŁÓWNA FUNKCJA DLA CIEBIE ---

def tile_image_with_uuid(image_path, output_folder, tile_size=256, overlap_ratio=0.5):
    """
    To jest ta JEDNA funkcja, której używasz.
    1. Wczytuje plik.
    2. Generuje UUID.
    3. Robi detekcję, crop, clahe, tiling i padding.
    """
    
    # 1. Przygotowanie danych pliku
    if not os.path.exists(image_path):
        print(f"BŁĄD: Nie znaleziono pliku {image_path}")
        return

    # Generujemy krótki UUID (8 znaków wystarczy, żeby było unikalne w folderze)
    file_uuid = str(uuid.uuid4())[:8]
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # 2. Wczytanie (Grayscale)
    full_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if full_img is None:
        print("BŁĄD: OpenCV nie może otworzyć pliku.")
        return

    # 3. Detekcja obiektu
    coords = _get_coordinates(full_img)
    if not coords:
        print(f"SKIP: Nie wykryto obiektu na {base_name}")
        return

    # 4. Wycięcie i Hardcore Processing
    x1, y1, x2, y2 = coords
    cropped_img = full_img[y1:y2, x1:x2]
    processed_img = _apply_hardcore_processing(cropped_img)

    # 5. Tiling Loop
    os.makedirs(output_folder, exist_ok=True)
    
    h, w = processed_img.shape
    stride = int(tile_size * (1 - overlap_ratio))
    stride = max(1, stride)
    
    count = 0
    
    print(f"Przetwarzanie: {base_name} (UUID: {file_uuid}) -> Wymiary: {w}x{h}")

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            tile = processed_img[y:y+tile_size, x:x+tile_size]
            t_h, t_w = tile.shape

            # Padding (BIAŁE TŁO 255)
            if t_h < tile_size or t_w < tile_size:
                # Tworzymy białą macierz 256x256
                canvas = np.full((tile_size, tile_size), 255, dtype=np.uint8)
                # Wklejamy wycinek
                canvas[0:t_h, 0:t_w] = tile
                final_tile = canvas
            else:
                final_tile = tile

            # 6. Zapis z UUID
            # Format: nazwaOryginalna__UUID__tile_X.bmp
            filename = f"{base_name}__{file_uuid}__tile_{count}.bmp"
            save_path = os.path.join(output_folder, filename)
            
            cv2.imwrite(save_path, final_tile)
            count += 1

    print(f" -> Zakończono. Wygenerowano {count} kafelków.")

# --- PRZYKŁAD UŻYCIA ---
if __name__ == "__main__":
    # Podajesz tylko plik i folder wyjściowy
    plik = r"C:\Users\jakub\Desktop\NAUKA\czyste\202511180023\48001F003202511180023 czarno.bmp"
    
    tile_image_with_uuid(plik, "dataset_z_uuid")