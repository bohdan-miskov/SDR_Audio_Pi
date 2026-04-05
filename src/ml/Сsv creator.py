import pandas as pd
from pathlib import Path


def generate_csv():
    current_dir = Path(__file__).resolve().parent

    project_root = current_dir.parent.parent

    audio_dir = project_root / 'Dataset' / 'Audio'

    csv_path = current_dir / 'drone_dataset.csv'

    data = []
    valid_extensions = {'.wav', '.flac'}

    print(f"Сканую папку: {audio_dir}...")

    if not audio_dir.exists():
        print(f"Помилка: Папка {audio_dir} не знайдена!")
        return

    for class_folder in audio_dir.iterdir():
        if class_folder.is_dir():
            label = class_folder.name

            for audio_file in class_folder.iterdir():
                if audio_file.suffix.lower() in valid_extensions:
                    data.append({
                        'fname': audio_file.name,
                        'label': label
                    })

    if not data:
        print("Жодного аудіофайлу не знайдено.")
        return

    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)

    print(f"Готово! Знайдено {len(df)} файлів.")
    print(f"Таблицю збережено тут: {csv_path.name}")
    print("-" * 20)
    print(df['label'].value_counts())


if __name__ == '__main__':
    generate_csv()