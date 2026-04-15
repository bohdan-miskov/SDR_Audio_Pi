import sys
import shutil
import soundfile as sf
from pathlib import Path
from src.ml.detect import DroneDetector


class Config: pass


setattr(sys.modules['__main__'], 'Config', Config)


def main():
    src_dir = Path(__file__).resolve().parent.parent

    m_path = src_dir / 'models' / 'conv.keras'
    p_path = src_dir / 'models' / 'conv.p'

    detector = DroneDetector(model_path=str(m_path), pickle_path=str(p_path))

    input_folder = src_dir / 'data' / 'input_audio'
    output_folder = src_dir / 'data' / 'recorded_alerts'

    input_folder.mkdir(parents=True, exist_ok=True)
    output_folder.mkdir(parents=True, exist_ok=True)

    print(f"Сканую: {input_folder}\n" + "-" * 40)

    valid_extensions = {'.wav', '.flac'}

    for filepath in input_folder.iterdir():
        if filepath.suffix.lower() not in valid_extensions:
            continue

        try:
            signal, rate = sf.read(str(filepath), dtype='float32')

            if len(signal.shape) > 1:
                signal = signal[:, 0]

            result = detector.predict(signal, rate=rate)

            if result['class'] == 'drone' and result['confidence'] > 70.0:
                print(f"ДРОН ({result['confidence']:>5}%): {filepath.name} -> Скопійовано")
                shutil.copy(str(filepath), str(output_folder / filepath.name))
            else:
                print(f"ШУМ  ({result['confidence']:>5}%): {filepath.name}")

        except Exception as e:
            print(f"Помилка з {filepath.name}: {e}")


if __name__ == "__main__":
    main()