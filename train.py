from anomalib.data import Folder
from anomalib.models import Patchcore
from anomalib.engine import Engine

def main():
    # 1. Konfiguracja Danych
    datamodule = Folder(
        name="my_custom_dataset",
        root="datasets",
        normal_dir="train/good",   # Folder treningowy
        abnormal_dir="test/defect", # Folder z defektami do testów
        normal_test_dir="test/good", # Folder z dobrymi próbkami do testów
        train_batch_size=32,
        eval_batch_size=32,
        # WAŻNE NA WINDOWS: Czasem warto ustawić num_workers na 0, jeśli nadal będą błędy
        # num_workers=0
    )

    # setup() też warto wywołać wewnątrz main
    datamodule.setup()

    # 2. Inicjalizacja Modelu
    model = Patchcore(
        backbone="resnet18",
        pre_trained=True
    )

    # 3. Konfiguracja Silnika (Engine)
    engine = Engine(
        accelerator="auto",
        max_epochs=1,
    )

    # 4. Trening
    print("Rozpoczynam trening...")
    engine.fit(datamodule=datamodule, model=model)

    # 5. Testowanie
    print("Rozpoczynam testy...")
    # Tutaj poprawka z poprzedniej odpowiedzi (setup dla testu), aby uniknąć błędu iter()
    datamodule.setup(stage="test")
    test_results = engine.test(datamodule=datamodule, model=model)
    print(test_results)

if __name__ == "__main__":
    # Ten blok jest KLUCZOWY na Windowsie
    main()