import os
import shutil
import warnings

# Ignorowanie ostrzeżeń
warnings.filterwarnings("ignore")

# --- FIX NA SYMLINKI WINDOWS ---
def fixed_symlink(target, link_name, target_is_directory=False):
    if target_is_directory:
        if os.path.exists(link_name):
            if os.path.islink(link_name) or os.path.isfile(link_name):
                os.remove(link_name)
            else:
                shutil.rmtree(link_name)
    pass

os.symlink = fixed_symlink
# -------------------------------

from anomalib.data import Folder
from anomalib.models import Patchcore
from anomalib.engine import Engine
from anomalib.deploy import ExportType

def train_main():
    print("--- KONFIGURACJA ---")
    
    # 1. Dataset
    # UWAGA: Poprawiłem ścieżki.
    # Jeśli Twoje kafelki są w folderze "dataset/normal_tiled" obok skryptu:
    datamodule = Folder(
        name="contraband_xray",
        root="../dataset",             # Folder główny
        normal_dir="normal_tiled",  # Podfolder z dobrymi (wewnątrz root)
        abnormal_dir="abnormal_tiled", # Podfolder ze złymi (wewnątrz root)
        #image_size=(256, 256),
        #train_batch_size=32,
        #eval_batch_size=32,
        num_workers=0               # Windows fix
    )

    # 2. Model
    model = Patchcore(
        backbone="wide_resnet50_2", 
        pre_trained=True,
        coreset_sampling_ratio=0.1, 
    )

    # 3. Engine
    # POPRAWKA: Usunięto 'task="classification"' z tego miejsca
    engine = Engine(
        accelerator="auto",
        devices=1,
        max_epochs=1,
        default_root_dir="results"
    )

    # 4. Trening
    print("\n--- ROZPOCZYNAM TRENING (Extracting Features) ---")
    engine.fit(datamodule=datamodule, model=model)

    # 5. Testowanie
    print("\n--- ROZPOCZYNAM TESTOWANIE ---")
    try:
        engine.test(datamodule=datamodule, model=model)
    except Exception as e:
        print(f"Ostrzeżenie przy testowaniu: {e}")

    # 6. Eksport
    print("\n--- EKSPORTOWANIE MODELU ---")
    try:
        openvino_path = engine.export(
            model=model,
            export_type=ExportType.OPENVINO,
            input_size=(256, 256),
            onnx_kwargs={"dynamic_axes": None} 
        )
        print(f"Model OpenVINO zapisany w: {openvino_path}")
        
    except Exception as e:
        print(f"Błąd eksportu OpenVINO: {e}")
        print("Próbuję eksportu do czystego Torch (.pt)...")
        engine.export(model=model, export_type=ExportType.TORCH)

    print("\n--- GOTOWE! ---")

if __name__ == "__main__":
    train_main()