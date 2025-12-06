import os
# Importujemy Twoją funkcję z pliku crop_image.py
from crop_image import tile_image_with_uuid

def prepare_training_data(input_path, output_path):
    """
    Przeszukuje input_path pod kątem obrazów i dla każdego wywołuje
    funkcję tile_image_with_uuid, zapisując wyniki w output_path.
    """
    
    # Obsługiwane formaty (dodaj inne jeśli masz np. tif)
    valid_extensions = ('.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff')
    
    # Licznik dla statystyki
    processed_count = 0

    print(f"--- START: Przetwarzanie folderu '{input_path}' ---")

    # os.walk pozwala wejść głęboko w podfoldery (recursive)
    for root, dirs, files in os.walk(input_path):
        for file in files:
            # Sprawdzamy czy to obrazek
            if file.lower().endswith(valid_extensions):
                full_path = os.path.join(root, file)
                
                try:
                    # WYWOŁANIE TWOJEJ FUNKCJI
                    # Możesz tu sterować parametrami (tile_size, overlap) globalnie
                    tile_image_with_uuid(
                        image_path=full_path, 
                        output_folder=output_path,
                        tile_size=256,      # Rozmiar kafelka
                        overlap_ratio=0.5   # Zakładka (50% to dobry standard)
                    )
                    processed_count += 1
                    
                except Exception as e:
                    print(f"!!! BŁĄD przy pliku {file}: {e}")

    print(f"--- KONIEC. Przetworzono {processed_count} plików źródłowych. ---")
    print(f"Wszystkie kafelki znajdują się w: {output_path}")

# --- PRZYKŁAD UŻYCIA ---
if __name__ == "__main__":
    # 1. Ścieżka do folderu z surowymi zdjęciami RTG (np. normalne)
    path_in = r"../dataset/abnormal"
    
    # 2. Gdzie mają trafić kafelki (to podasz potem do Anomalib)
    path_out = "../dataset/abnormal_tiled"
    
    prepare_training_data(path_in, path_out)