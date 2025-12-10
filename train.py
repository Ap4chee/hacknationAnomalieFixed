from anomalib.data import Folder
from anomalib.models import Patchcore
from anomalib.engine import Engine
from anomalib.deploy import ExportType
import os

def count_images(folder):
    """Policz obrazy w folderze."""
    if not os.path.exists(folder):
        return 0
    extensions = ('.bmp', '.png', '.jpg', '.jpeg')
    return len([f for f in os.listdir(folder) if f.lower().endswith(extensions)])

def train_main():
    print("--- KONFIGURACJA ---")
    
    print(f"Plików w 'good': {count_images('good')}")
    print(f"Plików w 'bad': {count_images('bad')}")
    
    datamodule = Folder(
        name="contraband_xray",
        root=".",
        normal_dir="fixed_data/good",
        #abnormal_dir="bad",
        train_batch_size=8,
        eval_batch_size=8,
        num_workers=0,              # Windows fix
        normal_split_ratio=0.9,     # 90% train
    )
    
    datamodule.setup()
    print(f"Train samples: {len(datamodule.train_data)}")
    print(f"Val samples: {len(datamodule.val_data)}")
    print(f"Test samples: {len(datamodule.test_data)}")

    model = Patchcore(
        backbone="resnet18",
        pre_trained=True,
        coreset_sampling_ratio=0.01, # 10% dla lepszej jakości
    )

    engine = Engine(
        accelerator="cpu",          # Jawnie CPU
        devices=1,
        max_epochs=1,
        default_root_dir="results"
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
        torch_path = engine.export(
            model=model,
            export_type=ExportType.TORCH,
        )
        print(f"Model TORCH zapisany w: {torch_path}")
    except Exception as e:
        print(f"Błąd eksportu: {e}")

    print("\n--- GOTOWE! ---")

if __name__ == "__main__":
    train_main()
