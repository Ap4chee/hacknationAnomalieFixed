from anomalib.data import Folder
from anomalib.models import Patchcore
from anomalib.engine import Engine
from anomalib.deploy import ExportType

def train_main():
    print("--- KONFIGURACJA ---")
    datamodule = Folder(
        name="contraband_xray", 			# nazwa eksperymentu
        root=".",             				# Folder główny
        normal_dir="normal",  				# Podfolder z dobrymi (wewnątrz root)
        #abnormal_dir="cropped_images_bad", 	# Podfolder ze złymi (wewnątrz root)
    )

    model = Patchcore(
        backbone="resnet18",				# Model do ekstrakcji cech z obrazów
        pre_trained=True,					# Użyj wstępnie wytrenowanych wag
        coreset_sampling_ratio=0.01,		# Procent próbek do zachowania 0.01 = 1%
    )

    engine = Engine(
        accelerator="auto",					# Lightning automatycznie wybiera GPU/MPS/CPU
        devices=1,							# Liczba urządzeń (GPU/MPS/CPU)
        max_epochs=1,						# Maksymalna liczba epok treningu
        default_root_dir="results"			# Katalog do zapisywania wyników
    )

    print("\n--- ROZPOCZYNAM TRENING (Extracting Features) ---")
    engine.fit(model=model, datamodule=datamodule)


    print("\n--- ROZPOCZYNAM TESTOWANIE ---")
    try:
        engine.test(datamodule=datamodule, model=model)
    except Exception as e:
        print(f"Ostrzeżenie przy testowaniu: {e}")


    print("\n--- EKSPORTOWANIE MODELU ---")
    try:
        openvino_path = engine.export(
            model=model,
            export_type=ExportType.TORCH,
        )
        print(f"Model TORCH zapisany w: {openvino_path}")

    except Exception as e:
        print(f"Błąd eksportu TORCH: {e}")
        engine.export(model=model, export_type=ExportType.OPENVINO)

    print("\n--- GOTOWE! ---")

if __name__ == "__main__":
    train_main()
