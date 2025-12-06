from PIL import Image
import os
import cv2


class OverlappingTiledCropper:
    def __init__(
        self,
        input_image_path,
        output_folder,
        crop_width=256,
        crop_height=256,
        min_overlap=0.5,
        to_grayscale=True,
        crop_to_object=True,   # <— control whether we first crop to detected object
    ):
        self.input_image_path = input_image_path
        self.output_folder = output_folder
        self.crop_width = crop_width
        self.crop_height = crop_height
        self.min_overlap = min_overlap
        self.to_grayscale = to_grayscale
        self.crop_to_object_flag = crop_to_object

        self.image = None
        self.width = None
        self.height = None

        self._load_and_prepare_image()

    # ----------------- IMAGE PREPARATION -----------------

    def _load_and_prepare_image(self):
        """Load image, optionally crop to detected object, then (optionally) convert to grayscale."""
        img = Image.open(self.input_image_path)

        # 1) Crop to detected object (using OpenCV) if requested
        if self.crop_to_object_flag:
            coords = self.get_object_coordinates()
            if coords is not None:
                img = img.crop(coords)
            else:
                print("Warning: no object detected, using full image.")

        # 2) Convert to grayscale if requested
        if self.to_grayscale:
            img = img.convert("L")

        self.image = img
        self.width, self.height = img.size

    # ----------------- TILING LOGIC -----------------

    def _compute_positions(self, total_size, crop_size):
        """
        Compute start positions along one axis to:
        - have at least min_overlap between neighbouring crops
        - cover the entire axis (0..total_size)
        """
        stride = int(crop_size * (1 - self.min_overlap))
        stride = max(1, stride)  # avoid zero stride

        # First pass: regular strides
        positions = list(range(0, max(1, total_size - crop_size + 1), stride))

        # Ensure last window touches the end
        last_start = total_size - crop_size
        if positions[-1] != last_start:
            positions.append(last_start)

        positions = sorted(set(positions))
        return positions

    def generate_tiled_crops(self):
        """Generate overlapping tiled crops covering 100% of the prepared image."""
        os.makedirs(self.output_folder, exist_ok=True)

        xs = self._compute_positions(self.width, self.crop_width)
        ys = self._compute_positions(self.height, self.crop_height)

        idx = 1
        for top in ys:
            bottom = top + self.crop_height
            for left in xs:
                right = left + self.crop_width

                crop = self.image.crop((left, top, right, bottom))
                out_path = os.path.join(
                    self.output_folder,
                    f"crop_{idx:04d}_x{left}_y{top}.bmp"
                )
                crop.save(out_path)
                idx += 1

        print(f"Saved {idx - 1} crops to '{self.output_folder}'.")

    # ----------------- OBJECT DETECTION -----------------

    def get_object_coordinates(self, margin=20):
        """
        Detect main object via thresholding and contours with OpenCV.
        Returns bounding box (x_start, y_start, x_end, y_end) in original image coordinates.
        """
        # Read as grayscale so THRESH_OTSU works
        img = cv2.imread(self.input_image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Error: cannot read image '{self.input_image_path}' with OpenCV.")
            return None

        # Blur to reduce noise
        blur = cv2.GaussianBlur(img, (5, 5), 0)

        # Assume bright background, dark object -> THRESH_BINARY_INV
        _, thresh = cv2.threshold(
            blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            print("No contours found.")
            return None

        # Largest contour = main object
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)

        h_img, w_img = img.shape[:2]

        x_start = max(x - margin, 0)
        y_start = max(y - margin, 0)
        x_end = min(x + w + margin, w_img)
        y_end = min(y + h + margin, h_img)

        return (x_start, y_start, x_end, y_end)


if __name__ == "__main__":
<<<<<<< HEAD
    plik_wejsciowy = r"C:\Users\gierc\PycharmProjects\PythonProject\brudne\202511190032\48001F003202511190032.bmp"
    folder_wyjsciowy = "test_tiled"
    
    print("Szukam obiektu...")
    coordinates = get_object_coordinates(plik_wejsciowy)
    
    if coordinates:
        process_with_pil(plik_wejsciowy, folder_wyjsciowy, coordinates, overlap_ratio=0.5)
    else:
        print("Błąd detekcji.")
=======
    input_image_path = "NAUKA/czyste/202511180021/48001F003202511180021.bmp"
    output_folder = "cropped_images"

    cropper = OverlappingTiledCropper(
        input_image_path=input_image_path,
        output_folder=output_folder,
        crop_width=256,
        crop_height=256,
        min_overlap=0.5,    # 50% overlap
        to_grayscale=True,
        crop_to_object=True
    )

    cropper.generate_tiled_crops()
>>>>>>> 60c967f7ce03c91d8e6b1c2730093d8f592246f6
