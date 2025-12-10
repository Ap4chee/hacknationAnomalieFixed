import os
from ImageProcessor.XRayImage import XRayImage

def normalize(data_folder: str, output_directory: str):

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    if not os.path.exists(data_folder):
        print(f"Folder ze zdjęciami do przetworzenia nie istnieje: {data_folder}")
        exit(1)


    for root, dirs, files in os.walk(data_folder):
        for file in files:
            if file.find('czarno') == -1:
                continue

            # Sprawdzamy czy to obrazek
            if file.lower().endswith('.bmp'):
                full_path = os.path.join(root, file)

                img = XRayImage(src=full_path)
                img.applyFilters()
                img.generateTiles()
                img.saveTiles(output_directory)


if __name__ == "__main__":
    normalize("raw_data/czyste", "normalized_data/good")
    normalize("raw_data/brudne", "normalized_data/bad")