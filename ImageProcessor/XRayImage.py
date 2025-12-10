import cv2
import numpy as np
import os
from .XRayFilters import XRayFilters

class Tile:
    """
    Reprezentuje jeden kafelek wycięty z obrazu.
    """
    def __init__(self, index, x, y, data):
        self.index = index
        self.x = x
        self.y = y
        self.data = data

    def __repr__(self):
        return f"Tile(index={self.index}, x={self.x}, y={self.y}, shape={self.data.shape})"

    def save(self, output_dir, base_name="tile"):
        """
        Zapisuje kafelek do katalogu.
        """
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, f"{base_name}_{self.index}_{self.x}_{self.y}.bmp")
        cv2.imwrite(file_path, self.data)
        return file_path


class XRayImage:
    """
    Reprezentacja obrazu RTG z obsługą:
        - wczytania
        - filtracji
        - generowania kafelków
        - zapisywania kafelków
    """

    def __init__(self, src=None):
        self.imgPath = src
        self.img = None
        self.tiles = []
        self.filters = XRayFilters()
        if src is not None:
            self.load(src)

    # ------------------------------------------------------------------
    def load(self, src: str) -> None:
        if not os.path.exists(src):
            raise FileNotFoundError(f"File not found: {src}")
        
        self.img = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
        
        if self.img is None:
            raise ValueError(f"Failed to load image: {src}")
        
        # Dodatkowa walidacja
        if self.img.size == 0:
            raise ValueError(f"Loaded image is empty: {src}")
        
        if self.img.shape[0] < 10 or self.img.shape[1] < 10:
            raise ValueError(f"Image too small: {self.img.shape}")
        
        self.imgPath = src

    # ------------------------------------------------------------------
    def applyFilters(self):
        if self.img is None:
            raise RuntimeError("Nie można zastosować filtrów — obraz nie został wczytany.")
        self.img = self.filters.process(self.img)

    # ------------------------------------------------------------------
    def generateTiles(self, tile_size=256, min_overlap=0.5, crop_to_object=False):
        """
        Tworzy nakładające się kafelki (tiles) z przetworzonego obrazu.

        Args:
            tile_size: Rozmiar jednego kafelka (kwadrat).
            min_overlap: Minimalny procent nakładania się sąsiednich kafelków (0..1).
            crop_to_object: Jeśli True, przycina obraz do wykrytego obiektu przed kafelkowaniem.
        """
        if self.img is None:
            raise RuntimeError("Najpierw wczytaj i przetwórz obraz, zanim wygenerujesz tiles.")

        img_to_tile = self.img

        # ------------------ Opcjonalne crop do wykrytego obiektu ------------------
        if crop_to_object:
            coords = self.filters.get_object_coordinates(img_to_tile)
            if coords is not None:
                x_start, y_start, x_end, y_end = coords
                img_to_tile = img_to_tile[y_start:y_end, x_start:x_end]
            else:
                print("Warning: nie znaleziono obiektu, używam pełnego obrazu.")

        h, w = img_to_tile.shape
        stride = int(tile_size * (1 - min_overlap))
        stride = max(1, stride)

         # Funkcja pomocnicza dla osi (x lub y)
        def compute_positions(total_size):
            positions = list(range(0, max(1, total_size - tile_size + 1), stride))
            last_start = total_size - tile_size
            if positions[-1] != last_start:
                positions.append(last_start)
            return sorted(set(positions))

        xs = compute_positions(w)
        ys = compute_positions(h)

        self.tiles = []
        index = 0

        for y in ys:
            for x in xs:
                tile_data = img_to_tile[y:y + tile_size, x:x + tile_size].copy()

                # Pomijamy całkowicie/prawie białe kafelki (średnia > 240)
                # oraz kafelki gdzie >90% pikseli jest białych (>245)
                mean_val = np.mean(tile_data)
                white_ratio = np.sum(tile_data > 245) / tile_data.size
                
                if mean_val >= 240 or white_ratio > 0.90:
                    continue

                tile = Tile(index=index, x=x, y=y, data=tile_data)
                self.tiles.append(tile)
                index += 1

        return self.tiles


    # ------------------------------------------------------------------
    def saveTiles(self, output_dir="tiles_output"):
        """
        Zapisuje wszystkie wygenerowane kafelki do katalogu.
        """
        if not self.tiles:
            raise RuntimeError("Nie wygenerowano jeszcze żadnych tiles.")
        base_name = os.path.splitext(os.path.basename(self.imgPath))[0]
        saved_paths = []
        for tile in self.tiles:
            path = tile.save(output_dir, base_name)
            saved_paths.append(path)
        return saved_paths


# # ------------------------ TESTY ------------------------
# if __name__ == '__main__':
#     xray = XRayImage("C:\\Users\\jakub\\Desktop\\raw_data\\czyste\\202511180023\\48001F003202511180023 czarno.bmp")
#     xray.applyFilters()
#     tiles = xray.generateTiles(tile_size=512)

#     saved_paths = xray.saveTiles("tiles_test")
