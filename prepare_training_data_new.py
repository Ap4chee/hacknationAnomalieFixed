import os
from ImageProcessor.XRayImage import XRayImage

clean_data_folder = "NAUKA/czyste"
clean_output_directory = "normal"

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
