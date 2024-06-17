import csv
import pathlib
import shutil
import datetime
import tqdm

def main():
    transcriptions = []
    outputs_path = pathlib.Path("src/outputs")
    output_path = outputs_path / datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    output_path.mkdir()
    datasets_path = pathlib.Path("src/datasets")
    dataset_paths = list(datasets_path.iterdir())
    with tqdm.tqdm(total=len(dataset_paths)) as pbar:
        for dataset_path in dataset_paths:
            if not dataset_path.is_dir():
                continue
            with open(dataset_path / "transcriptions.csv", "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    transcriptions.append([
                        f"{row['name']}_{dataset_path.stem}",
                        row["ph_seq"],
                        row["ph_dur"]
                    ])
            for wav_path in (dataset_path / "wavs").glob("*.wav"):
                shutil.copy(wav_path, output_path / f"{wav_path.stem}_{dataset_path.stem}.wav")
            pbar.update(1)
    with open(output_path / "transcriptions.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "ph_seq", "ph_dur"])
        writer.writerows(transcriptions)
    print(f"Output saved in {output_path}")