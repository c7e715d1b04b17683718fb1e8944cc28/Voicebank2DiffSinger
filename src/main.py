import sys

sys.path.append("src/SOFA")
sys.path.append("src/SOFA/modules")
sys.path.append("src/MakeDiffSinger/acoustic_forced_alignment")
import pathlib
import tqdm
import re
from SOFA.modules.g2p.base_g2p import DataFrameDataset
import pandas as pd
import warnings
import pyopenjtalk
import SOFA.infer
import SOFA.modules.g2p
import SOFA.modules.AP_detector
import torch
from SOFA.train import LitForcedAlignmentTask
import lightning as pl
from click import Context
from MakeDiffSinger.acoustic_forced_alignment.build_dataset import build_dataset
import datetime
import shutil
import csv
from click import Command
import importlib.util


def import_module_from_path(module_path: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


add_ph_num: Command = import_module_from_path(
    "src/MakeDiffSinger/variance-temp-solution/add_ph_num.py", "add_ph_num"
).add_ph_num
estimate_midi: Command = import_module_from_path(
    "src/MakeDiffSinger/variance-temp-solution/estimate_midi.py", "estimate_midi"
).estimate_midi
csv2ds: Command = import_module_from_path(
    "src/MakeDiffSinger/variance-temp-solution/convert_ds.py", "convert_ds"
).csv2ds

VERSION = "0.0.1"
HIRAGANA_REGEX = re.compile(r"([あ-ん][ぁぃぅぇぉゃゅょ]|[あ-ん])")
KATAKANA_REGEX = re.compile(r"([ア-ン][ァィゥェォャュョ]|[ア-ン])")


def remove_specific_consecutive_duplicates(
    input_list: list[str], specific_elements: list[str]
):
    if not input_list:
        return []

    # Initialize the result list with the first element
    result = [input_list[0]]

    # Iterate through the input list starting from the second element
    for item in input_list[1:]:
        # If the current item is different from the last item in the result list
        # or if it is not in the specific elements list, add it
        if item != result[-1] or item not in specific_elements:
            result.append(item)

    return result


class PyOpenJTalkG2P:
    def __call__(self, text: str):
        ph_seq, word_seq, ph_idx_to_word_idx = self._g2p(text)

        # The first and last phonemes should be `SP`,
        # and there should not be more than two consecutive `SP`s at any position.
        assert ph_seq[0] == "SP" and ph_seq[-1] == "SP"
        assert all(
            ph_seq[i] != "SP" or ph_seq[i + 1] != "SP" for i in range(len(ph_seq) - 1)
        )
        return ph_seq, word_seq, ph_idx_to_word_idx

    def _g2p(self, input_text: str):
        word_seq_raw = input_text.strip().split(" ")
        word_seq = []
        word_seq_idx = 0
        ph_seq = ["SP"]
        ph_idx_to_word_idx = [-1]
        for word in word_seq_raw:
            ph_raw = pyopenjtalk.g2p(word)
            if not ph_raw:
                warnings.warn(f"Word {word} is not in the dictionary. Ignored.")
                continue
            word_seq.append(word)
            phones = ph_raw.split(" ")
            for i, ph in enumerate(phones):
                if (i == 0 or i == len(phones) - 1) and ph == "SP":
                    warnings.warn(
                        f"The first or last phoneme of word {word} is SP, which is not allowed. "
                        "Please check your dictionary."
                    )
                    continue
                ph_seq.append(ph)
                ph_idx_to_word_idx.append(word_seq_idx)
            if ph_seq[-1] != "SP":
                ph_seq.append("SP")
                ph_idx_to_word_idx.append(-1)
            word_seq_idx += 1

        return ph_seq, word_seq, ph_idx_to_word_idx

    def get_dataset(self, wav_paths: list[pathlib.Path]):
        dataset = []
        for wav_path in wav_paths:
            try:
                if wav_path.with_suffix(".txt").exists():
                    with open(wav_path.with_suffix(".txt"), "r", encoding="utf-8") as f:
                        lab_text = f.read().strip()
                    ph_seq, word_seq, ph_idx_to_word_idx = self(lab_text)
                    dataset.append((wav_path, ph_seq, word_seq, ph_idx_to_word_idx))
            except Exception as e:
                e.args = (f" Error when processing {wav_path}: {e} ",)
                raise e
        dataset = pd.DataFrame(
            dataset, columns=["wav_path", "ph_seq", "word_seq", "ph_idx_to_word_idx"]
        )
        return DataFrameDataset(dataset)


def main():
    print(
        f"Voicebank to DiffSinger {VERSION} - Convert the UTAU sound source folder to a configuration compatible with DiffSinger"
    )
    print()

    voicebank_dirs = [pathlib.Path(voicebank_dir) for voicebank_dir in sys.argv[1:]]

    print("Phase 1: Generating text files...")
    print()

    with tqdm.tqdm(total=len(voicebank_dirs)) as pbar:
        for voicebank_folder_path in voicebank_dirs:
            voicebank_wav_files = list(voicebank_folder_path.glob("*.wav"))
            for wav_file in voicebank_wav_files:
                file_name = pathlib.Path(wav_file).stem
                words = file_name[1:]
                graphemes = remove_specific_consecutive_duplicates(
                    [
                        *HIRAGANA_REGEX.findall(words),
                        *KATAKANA_REGEX.findall(words),
                    ],
                    ["あ", "い", "う", "え", "お", "ん"],
                )
                with open(
                    str(voicebank_folder_path) + "/" + file_name + ".txt",
                    "w",
                    encoding="utf-8",
                ) as f:
                    f.write(" ".join(graphemes))
            pbar.update(1)

    print()
    print("Phase 1: Done.")
    print()
    print("Phase 2: Generating TextGrids...")
    print()

    AP_detector_class = SOFA.modules.AP_detector.LoudnessSpectralcentroidAPDetector
    get_AP = AP_detector_class()

    g2p_class = PyOpenJTalkG2P
    grapheme_to_phoneme = g2p_class()

    torch.set_grad_enabled(False)

    model = LitForcedAlignmentTask.load_from_checkpoint(
        "src/cktp/japanese-v2.0-45000.ckpt"
    )
    model.set_inference_mode("force")

    trainer = pl.Trainer(logger=False)

    for voicebank_folder_path in voicebank_dirs:
        voicebank_wav_files = list(voicebank_folder_path.glob("*.wav"))
        dataset = grapheme_to_phoneme.get_dataset(
            list(voicebank_folder_path.glob("*.wav"))
        )

        predictions = trainer.predict(
            model, dataloaders=dataset, return_predictions=True
        )

        predictions = get_AP.process(predictions)
        predictions = SOFA.infer.post_processing(predictions)

        SOFA.infer.save_textgrids(predictions)

    print()
    print("Phase 2: Done.")
    print()
    print("Phase 3: Build dataset...")
    print()

    for voicebank_folder_path in voicebank_dirs:
        ctx = Context(build_dataset)
        with ctx:
            build_dataset.parse_args(
                ctx,
                [
                    "--wavs",
                    str(voicebank_folder_path),
                    "--tg",
                    str(voicebank_folder_path / "TextGrid"),
                    "--dataset",
                    str(voicebank_folder_path / "Dataset"),
                ],
            )
            build_dataset.invoke(ctx)

    print()
    print("Phase 3: Done.")
    print()
    print("Phase 4: Merge datasets...")
    print()

    transcriptions = []
    outputs_path = pathlib.Path("src/outputs")
    output_path = outputs_path / datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    output_path.mkdir()
    output_wavs_path = output_path / "wavs"
    output_wavs_path.mkdir()
    with tqdm.tqdm(total=len(voicebank_dirs)) as pbar:
        for voicebank_folder_path in voicebank_dirs:
            with open(
                voicebank_folder_path / "Dataset" / "transcriptions.csv",
                "r",
                encoding="utf-8",
            ) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    transcriptions.append(
                        [
                            f"{row['name']}_{voicebank_folder_path.stem}",
                            row["ph_seq"],
                            row["ph_dur"],
                        ]
                    )
            for wav_path in (voicebank_folder_path / "Dataset" / "wavs").glob("*.wav"):
                shutil.copy(
                    wav_path,
                    output_wavs_path
                    / f"{wav_path.stem}_{voicebank_folder_path.stem}.wav",
                )
            pbar.update(1)
    with open(
        output_path / "transcriptions.csv", "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["name", "ph_seq", "ph_dur"])
        writer.writerows(transcriptions)

    print()
    print("Phase 4: Done.")
    print()
    print("Phase 5: Add phoneme number...")
    print()

    ctx = Context(add_ph_num)
    with ctx:
        add_ph_num.parse_args(
            ctx,
            [
                str(output_path / "transcriptions.csv"),
                "--dictionary",
                "src/dictionaries/japanese-dictionary.txt",
            ],
        )
        add_ph_num.invoke(ctx)

    print()
    print("Phase 5: Done.")
    print()
    print("Phase 6: Estimate MIDI...")
    print()

    ctx = Context(estimate_midi)
    with ctx:
        estimate_midi.parse_args(
            ctx,
            [
                str(output_path / "transcriptions.csv"),
                str(output_wavs_path),
            ],
        )
        estimate_midi.invoke(ctx)

    print()
    print("Phase 6: Done.")
    print()
    print("Phase 7: Convert CSV to DiffSinger...")
    print()

    ctx = Context(csv2ds)
    with ctx:
        csv2ds.parse_args(
            ctx,
            [
                str(output_path / "transcriptions.csv"),
                str(output_wavs_path),
            ],
        )
        csv2ds.invoke(ctx)

    print()
    print("Phase 7: Done.")
    print()
    input("Press Enter to exit...")


if __name__ == "__main__":
    main()
