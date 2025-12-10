import os
from ImageProcessor.XRayImage import XRayImage


if __name__ == "__main__":
    if os.sys.argv and len(os.sys.argv) == 3:
        clean_data_folder = os.sys.argv[1]
        clean_output_directory = os.sys.argv[2]
    else:
        clean_data_folder = "raw_data/czyste"
        clean_output_directory = "good"
        if not os.path.exists(clean_output_directory):
            os.makedirs(clean_output_directory)
        if not os.path.exists(clean_data_folder):
            print(f"Folder ze zdjęciami do przetworzenia nie istnieje: {clean_data_folder}")
            exit(1)


    for root, dirs, files in os.walk(clean_data_folder):
        for file in files:
            if file.find('czarno') == -1:
                continue

            # Sprawdzamy czy to obrazek
            if file.lower().endswith('.bmp'):
                full_path = os.path.join(root, file)

                img = XRayImage(src=full_path)
                img.applyFilters()
                img.generateTiles()
                img.saveTiles(clean_output_directory)
