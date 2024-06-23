import sys

sys.path.append("src/SOFA")
sys.path.append("src/SOFA/modules")
sys.path.append("src/MakeDiffSinger/acoustic_forced_alignment")
sys.path.append("src/MakeDiffSinger/variance-temp-solution")
import tempfile
import pathlib
import tqdm
import re
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
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
from click import Command
import importlib.util
import requests
from bs4 import BeautifulSoup
import subprocess
import utaupy
import time
import textgrid


if not pathlib.Path("src/Moresampler").exists():
    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = pathlib.Path(temp_dir_str)
        file_id = 139123
        file_name = pathlib.Path("Moresampler.zip")
        with requests.Session() as session:
            session.headers.update(
                {
                    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
                }
            )
            response = session.get(f"https://bowlroll.net/file/{file_id}")
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            data = {
                "download_key": "bowlroll_download_control_mischievous",
                "csrf_token": soup.find("div", {"id": "initialize"})["data-csrf_token"],
            }
            response = session.post(
                f"https://bowlroll.net/api/file/{file_id}/download-check", data=data
            )
            response.raise_for_status()
            response = session.get(response.json()["url"], stream=True)
            response.raise_for_status()
            with open(temp_dir / file_name, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        shutil.unpack_archive(temp_dir / file_name, temp_dir / "Moresampler")
        shutil.move(temp_dir / "Moresampler", "src/Moresampler")


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


def remove_duplicate_otos(otos: list[utaupy.otoini.Oto]):
    unique_otos: list[utaupy.otoini.Oto] = []
    for oto in otos:
        for unique_oto in unique_otos:
            if (
                oto.filename == unique_oto.filename
                and oto.offset == unique_oto.offset
                and oto.consonant == unique_oto.consonant
                and oto.cutoff == unique_oto.cutoff
                and oto.preutterance == unique_oto.preutterance
                and oto.overlap == unique_oto.overlap
            ):
                break
        else:
            unique_otos.append(oto)
    return unique_otos


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
            phones = pyopenjtalk.g2p(word, join=False)
            if not phones:
                warnings.warn(f"Word {word} is not in the dictionary. Ignored.")
                continue
            word_seq.append(word)
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
    if len(sys.argv) < 2:
        print("Usage: python src/main.py [voicebank_dir1] [voicebank_dir2] ...")
        sys.exit(1)
    print("Select the forced aligner to use:")
    print("1: SOFA")
    print("2: Moresampler")
    forced_aligner_type = input("Enter the number of the forced aligner to use: ")
    if forced_aligner_type == "1":
        forced_aligner = "SOFA"
    elif forced_aligner_type == "2":
        forced_aligner = "Moresampler"
    else:
        print("Invalid input.")
        sys.exit(1)
    detect_nonslicent_flag = input("Do you want to detect nonsilence? (y/n): ") == "y"
    print()

    voicebank_dirs = [pathlib.Path(voicebank_dir) for voicebank_dir in sys.argv[1:]]

    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = pathlib.Path(temp_dir_str)
        if forced_aligner == "SOFA":
            print("Phase 1: Merge voicebanks...")
            print()

            with tqdm.tqdm(total=len(voicebank_dirs)) as pbar:
                for voicebank_dir in voicebank_dirs:
                    for wav_file in voicebank_dir.glob("*.wav"):
                        shutil.copy(
                            wav_file,
                            temp_dir / f"{wav_file.stem}_{voicebank_dir.stem}.wav",
                        )
                    pbar.update(1)

            wav_files = list(temp_dir.glob("*.wav"))

            print()
            print("Phase 1: Done.")
            print()

            if detect_nonslicent_flag:
                print("Phase 1-1: Detecting nonsilence...")
                print()

                with tqdm.tqdm(total=len(wav_files)) as pbar:
                    for wav_file in wav_files:
                        audio: AudioSegment = AudioSegment.from_file(wav_file)
                        nonsilent_ranges = detect_nonsilent(
                            audio, min_silence_len=500, silence_thresh=-50
                        )
                        if nonsilent_ranges:
                            trimmed_audio = audio[
                                max(0, nonsilent_ranges[0][0] - 200) : min(
                                    len(audio), nonsilent_ranges[-1][1] + 200
                                )
                            ]
                            trimmed_audio.export(wav_file, format="wav")
                        pbar.update(1)

                print()
                print("Phase 1-1: Done.")
                print()

            print("Phase 2: Generating text files...")
            print()

            with tqdm.tqdm(total=len(wav_files)) as pbar:
                for wav_file in wav_files:
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
                        str(temp_dir) + "/" + file_name + ".txt",
                        "w",
                        encoding="utf-8",
                    ) as f:
                        f.write(" ".join(graphemes))
                    pbar.update(1)

            print()
            print("Phase 2: Done.")
            print()
            print("Phase 3: Generating TextGrids...")
            print()

            AP_detector_class = (
                SOFA.modules.AP_detector.LoudnessSpectralcentroidAPDetector
            )
            get_AP = AP_detector_class()

            g2p_class = PyOpenJTalkG2P
            grapheme_to_phoneme = g2p_class()

            torch.set_grad_enabled(False)

            model = LitForcedAlignmentTask.load_from_checkpoint(
                "src/cktp/japanese-v2.0-45000.ckpt"
            )
            model.set_inference_mode("force")

            trainer = pl.Trainer(logger=False)

            dataset = grapheme_to_phoneme.get_dataset(wav_files)

            predictions = trainer.predict(
                model, dataloaders=dataset, return_predictions=True
            )

            predictions = get_AP.process(predictions)
            predictions = SOFA.infer.post_processing(predictions)

            SOFA.infer.save_textgrids(predictions)

            print()
            print("Phase 3: Done.")
            print()
        elif forced_aligner == "Moresampler":
            print("Phase 1: Generating oto.ini file...")
            print()

            for voicebank_dir in voicebank_dirs:
                temp_voicebank_dir = temp_dir / voicebank_dir.stem
                temp_voicebank_dir.mkdir()
                wav_files = list(voicebank_dir.glob("*.wav"))
                with tqdm.tqdm(total=len(wav_files)) as pbar:
                    for wav_file in voicebank_dir.glob("*.wav"):
                        shutil.copy(
                            wav_file,
                            temp_voicebank_dir / wav_file.name,
                        )
                        pbar.update(1)
                print()
                process = subprocess.Popen(
                    [
                        "src/Moresampler/moresampler.exe",
                        str(temp_voicebank_dir),
                    ],
                    stdin=subprocess.PIPE,
                    text=True,
                )
                process.stdin.write("1\n")
                process.stdin.flush()
                process.stdin.write("y\n")
                process.stdin.flush()
                process.stdin.write("n\n")
                process.stdin.flush()
                process.stdin.write("1\n")
                process.stdin.flush()
                process.stdin.write("n\n")
                process.stdin.flush()
                process.stdin.write("\n")
                process.stdin.flush()
                while process.poll() is None:
                    process.stdin.write("\n")
                    process.stdin.flush()
                    time.sleep(0.1)
                print()

            print()
            print("Phase 1: Done.")
            print()
            print("Phase 2: Merge voicebanks...")
            print()

            merged_oto_ini = utaupy.otoini.OtoIni()
            with tqdm.tqdm(total=len(voicebank_dirs)) as pbar:
                for voicebank_dir in voicebank_dirs:
                    temp_voicebank_dir = temp_dir / voicebank_dir.stem
                    oto_ini = utaupy.otoini.load(str(temp_voicebank_dir / "oto.ini"))
                    for wav_file in (temp_voicebank_dir).glob("*.wav"):
                        shutil.move(
                            wav_file,
                            temp_dir / f"{wav_file.stem}_{voicebank_dir.stem}.wav",
                        )
                        otos: list[utaupy.otoini.Oto] = list(
                            filter(lambda oto: oto.filename == wav_file.name, oto_ini)
                        )
                        for oto in otos:
                            oto.filename = f"{wav_file.stem}_{voicebank_dir.stem}.wav"
                            merged_oto_ini.append(oto)
                    shutil.rmtree(temp_voicebank_dir)
                    pbar.update(1)
            merged_oto_ini.write(str(temp_dir / "oto.ini"))

            print()
            print("Phase 2: Done.")
            print()
            print("Phase 3: Convert oto.ini to TextGrid...")
            print()

            textgrid_dir = temp_dir / "TextGrid"
            textgrid_dir.mkdir()
            oto_ini = utaupy.otoini.load(str(temp_dir / "oto.ini"))
            wav_files = list(temp_dir.glob("*.wav"))
            with tqdm.tqdm(total=len(wav_files)) as pbar:
                for wav_file in temp_dir.glob("*.wav"):
                    otos: list[utaupy.otoini.Oto] = remove_duplicate_otos(
                        list(filter(lambda oto: oto.filename == wav_file.name, oto_ini))
                    )
                    sorted_otos = sorted(otos, key=lambda oto: oto.offset)
                    if any(
                        [
                            len(pyopenjtalk.g2p(oto.alias.split()[1], join=False)) > 2
                            for oto in sorted_otos
                            if oto.alias.split()[1] != "-"
                        ]
                    ):
                        wav_file.unlink()
                        pbar.update(1)
                        continue
                    audio = AudioSegment.from_file(wav_file)
                    tg = textgrid.TextGrid()
                    grapheme_tier = textgrid.IntervalTier(
                        name="graphemes", minTime=0, maxTime=audio.duration_seconds
                    )
                    phoneme_tier = textgrid.IntervalTier(
                        name="phonemes", minTime=0, maxTime=audio.duration_seconds
                    )
                    for i, oto in enumerate(sorted_otos[:-1]):
                        splitted_alias = oto.alias.split()
                        next_splitted_alias = sorted_otos[i + 1].alias.split()
                        phs = (
                            pyopenjtalk.g2p(splitted_alias[1], join=False)
                            if splitted_alias[1] != "-"
                            else []
                        )
                        next_phs = (
                            pyopenjtalk.g2p(next_splitted_alias[1], join=False)
                            if next_splitted_alias[1] != "-"
                            else []
                        )
                        if i == 0:
                            if len(next_phs) == 0:
                                if len(phs) == 1:
                                    # phoneme = utaupy.label.Phoneme()
                                    # phoneme.symbol = "SP"
                                    # phoneme.start = 0
                                    # phoneme.end = (oto.offset + oto.preutterance) * 1000
                                    # label.append(phoneme)
                                    # phoneme = utaupy.label.Phoneme()
                                    # phoneme.symbol = phs[0]
                                    # phoneme.start = (oto.offset + oto.preutterance) * 1000
                                    # phoneme.end = (otos[i + 1].offset + otos[i + 1].preutterance) * 1000
                                    # label.append(phoneme)
                                    # phoneme = utaupy.label.Phoneme()
                                    # phoneme.symbol = "SP"
                                    # phoneme.start = (otos[i + 1].offset + otos[i + 1].preutterance) * 1000
                                    # phoneme.end = audio_length
                                    # label.append(phoneme)
                                    grapheme_tier.add(
                                        0,
                                        (oto.offset + oto.preutterance) / 1000,
                                        "SP",
                                    )
                                    grapheme_tier.add(
                                        (oto.offset + oto.preutterance) / 1000,
                                        audio.duration_seconds,
                                        splitted_alias[1],
                                    )
                                    grapheme_tier.add(
                                        0,
                                        (oto.offset + oto.preutterance) / 1000,
                                        "SP",
                                    )
                                    phoneme_tier.add(
                                        0,
                                        (oto.offset + oto.preutterance) / 1000,
                                        "SP",
                                    )
                                    phoneme_tier.add(
                                        (oto.offset + oto.preutterance) / 1000,
                                        (otos[i + 1].offset + otos[i + 1].preutterance)
                                        / 1000,
                                        phs[0],
                                    )
                                    phoneme_tier.add(
                                        (otos[i + 1].offset + otos[i + 1].preutterance)
                                        / 1000,
                                        audio.duration_seconds,
                                        "SP",
                                    )
                                elif len(phs) == 2:
                                    # phoneme = utaupy.label.Phoneme()
                                    # phoneme.symbol = "SP"
                                    # phoneme.start = 0
                                    # phoneme.end = (oto.offset + oto.overlap) * 1000
                                    # label.append(phoneme)
                                    # phoneme = utaupy.label.Phoneme()
                                    # phoneme.symbol = phs[0]
                                    # phoneme.start = (oto.offset + oto.overlap) * 1000
                                    # phoneme.end = (oto.offset + oto.preutterance) * 1000
                                    # label.append(phoneme)
                                    # phoneme = utaupy.label.Phoneme()
                                    # phoneme.symbol = phs[1]
                                    # phoneme.start = (oto.offset + oto.preutterance) * 1000
                                    # phoneme.end = (otos[i + 1].offset + otos[i + 1].preutterance) * 1000
                                    # label.append(phoneme)
                                    # phoneme = utaupy.label.Phoneme()
                                    # phoneme.symbol = "SP"
                                    # phoneme.start = (otos[i + 1].offset + otos[i + 1].preutterance) * 1000
                                    # phoneme.end = audio_length
                                    # label.append(phoneme)
                                    grapheme_tier.add(
                                        0,
                                        (oto.offset + oto.overlap) / 1000,
                                        "SP",
                                    )
                                    grapheme_tier.add(
                                        (oto.offset + oto.overlap) / 1000,
                                        (otos[i + 1].offset + otos[i + 1].preutterance)
                                        / 1000,
                                        splitted_alias[1],
                                    )
                                    grapheme_tier.add(
                                        (otos[i + 1].offset + otos[i + 1].preutterance)
                                        / 1000,
                                        audio.duration_seconds,
                                        "SP",
                                    )
                                    phoneme_tier.add(
                                        0,
                                        (oto.offset + oto.overlap) / 1000,
                                        "SP",
                                    )
                                    phoneme_tier.add(
                                        (oto.offset + oto.overlap) / 1000,
                                        (oto.offset + oto.preutterance) / 1000,
                                        phs[0],
                                    )
                                    phoneme_tier.add(
                                        (oto.offset + oto.preutterance) / 1000,
                                        (otos[i + 1].offset + otos[i + 1].preutterance)
                                        / 1000,
                                        phs[1],
                                    )
                                    phoneme_tier.add(
                                        (otos[i + 1].offset + otos[i + 1].preutterance)
                                        / 1000,
                                        audio.duration_seconds,
                                        "SP",
                                    )
                                else:
                                    raise ValueError("Invalid phoneme length.")
                            elif len(next_phs) == 1:
                                if len(phs) == 1:
                                    # phoneme = utaupy.label.Phoneme()
                                    # phoneme.symbol = "SP"
                                    # phoneme.start = 0
                                    # phoneme.end = (oto.offset + oto.preutterance) * 1000
                                    # label.append(phoneme)
                                    # phoneme = utaupy.label.Phoneme()
                                    # phoneme.symbol = phs[0]
                                    # phoneme.start = (oto.offset + oto.preutterance) * 1000
                                    # phoneme.end = (otos[i + 1].offset + otos[i + 1].preutterance) * 1000
                                    # label.append(phoneme)
                                    grapheme_tier.add(
                                        0,
                                        (oto.offset + oto.preutterance) / 1000,
                                        "SP",
                                    )
                                    grapheme_tier.add(
                                        (oto.offset + oto.preutterance) / 1000,
                                        (otos[i + 1].offset + otos[i + 1].preutterance)
                                        / 1000,
                                        splitted_alias[1],
                                    )
                                    phoneme_tier.add(
                                        0,
                                        (oto.offset + oto.preutterance) / 1000,
                                        "SP",
                                    )
                                    phoneme_tier.add(
                                        (oto.offset + oto.preutterance) / 1000,
                                        (otos[i + 1].offset + otos[i + 1].preutterance)
                                        / 1000,
                                        phs[0],
                                    )
                                elif len(phs) == 2:
                                    # phoneme = utaupy.label.Phoneme()
                                    # phoneme.symbol = "SP"
                                    # phoneme.start = 0
                                    # phoneme.end = (oto.offset + oto.overlap) * 1000
                                    # label.append(phoneme)
                                    # phoneme = utaupy.label.Phoneme()
                                    # phoneme.symbol = phs[0]
                                    # phoneme.start = (oto.offset + oto.overlap) * 1000
                                    # phoneme.end = (oto.offset + oto.preutterance) * 1000
                                    # label.append(phoneme)
                                    # phoneme = utaupy.label.Phoneme()
                                    # phoneme.symbol = phs[1]
                                    # phoneme.start = (oto.offset + oto.preutterance) * 1000
                                    # phoneme.end = (otos[i + 1].offset + otos[i + 1].preutterance) * 1000
                                    # label.append(phoneme)
                                    grapheme_tier.add(
                                        0,
                                        (oto.offset + oto.overlap) / 1000,
                                        "SP",
                                    )
                                    grapheme_tier.add(
                                        (oto.offset + oto.overlap) / 1000,
                                        (otos[i + 1].offset + otos[i + 1].preutterance)
                                        / 1000,
                                        splitted_alias[1],
                                    )
                                    phoneme_tier.add(
                                        0,
                                        (oto.offset + oto.overlap) / 1000,
                                        "SP",
                                    )
                                    phoneme_tier.add(
                                        (oto.offset + oto.overlap) / 1000,
                                        (oto.offset + oto.preutterance) / 1000,
                                        phs[0],
                                    )
                                    phoneme_tier.add(
                                        (oto.offset + oto.preutterance) / 1000,
                                        (otos[i + 1].offset + otos[i + 1].preutterance)
                                        / 1000,
                                        phs[1],
                                    )
                                else:
                                    raise ValueError("Invalid phoneme length.")
                            elif len(next_phs) == 2:
                                if len(phs) == 1:
                                    # phoneme = utaupy.label.Phoneme()
                                    # phoneme.symbol = "SP"
                                    # phoneme.start = 0
                                    # phoneme.end = (oto.offset + oto.preutterance) * 1000
                                    # label.append(phoneme)
                                    # phoneme = utaupy.label.Phoneme()
                                    # phoneme.symbol = phs[0]
                                    # phoneme.start = (oto.offset + oto.preutterance) * 1000
                                    # phoneme.end = (otos[i + 1].offset + otos[i + 1].overlap) * 1000
                                    # label.append(phoneme)
                                    grapheme_tier.add(
                                        0,
                                        (oto.offset + oto.preutterance) / 1000,
                                        "SP",
                                    )
                                    grapheme_tier.add(
                                        (oto.offset + oto.preutterance) / 1000,
                                        (otos[i + 1].offset + otos[i + 1].overlap)
                                        / 1000,
                                        splitted_alias[1],
                                    )
                                    phoneme_tier.add(
                                        0,
                                        (oto.offset + oto.preutterance) / 1000,
                                        "SP",
                                    )
                                    phoneme_tier.add(
                                        (oto.offset + oto.preutterance) / 1000,
                                        (otos[i + 1].offset + otos[i + 1].overlap)
                                        / 1000,
                                        phs[0],
                                    )
                                elif len(phs) == 2:
                                    # phoneme = utaupy.label.Phoneme()
                                    # phoneme.symbol = "SP"
                                    # phoneme.start = 0
                                    # phoneme.end = (oto.offset + oto.overlap) * 1000
                                    # label.append(phoneme)
                                    # phoneme = utaupy.label.Phoneme()
                                    # phoneme.symbol = phs[0]
                                    # phoneme.start = (oto.offset + oto.overlap) * 1000
                                    # phoneme.end = (oto.offset + oto.preutterance) * 1000
                                    # label.append(phoneme)
                                    # phoneme = utaupy.label.Phoneme()
                                    # phoneme.symbol = phs[1]
                                    # phoneme.start = (oto.offset + oto.preutterance) * 1000
                                    # phoneme.end = (otos[i + 1].offset + otos[i + 1].overlap) * 1000
                                    # label.append(phoneme)
                                    grapheme_tier.add(
                                        0,
                                        (oto.offset + oto.overlap) / 1000,
                                        "SP",
                                    )
                                    grapheme_tier.add(
                                        (oto.offset + oto.overlap) / 1000,
                                        (otos[i + 1].offset + otos[i + 1].overlap)
                                        / 1000,
                                        splitted_alias[1],
                                    )
                                    phoneme_tier.add(
                                        0,
                                        (oto.offset + oto.overlap) / 1000,
                                        "SP",
                                    )
                                    phoneme_tier.add(
                                        (oto.offset + oto.overlap) / 1000,
                                        (oto.offset + oto.preutterance) / 1000,
                                        phs[0],
                                    )
                                    phoneme_tier.add(
                                        (oto.offset + oto.preutterance) / 1000,
                                        (otos[i + 1].offset + otos[i + 1].overlap)
                                        / 1000,
                                        phs[1],
                                    )
                                else:
                                    raise ValueError("Invalid phoneme length.")
                            else:
                                raise ValueError("Invalid phoneme length.")
                        else:
                            if len(next_phs) == 0:
                                if len(phs) == 1:
                                    # phoneme = utaupy.label.Phoneme()
                                    # phoneme.symbol = phs[0]
                                    # phoneme.start = (oto.offset + oto.preutterance) * 1000
                                    # phoneme.end = (otos[i + 1].offset + otos[i + 1].preutterance) * 1000
                                    # label.append(phoneme)
                                    # phoneme = utaupy.label.Phoneme()
                                    # phoneme.symbol = "SP"
                                    # phoneme.start = (otos[i + 1].offset + otos[i + 1].preutterance) * 1000
                                    # phoneme.end = audio_length
                                    # label.append(phoneme)
                                    grapheme_tier.add(
                                        (oto.offset + oto.preutterance) / 1000,
                                        (otos[i + 1].offset + otos[i + 1].preutterance)
                                        / 1000,
                                        splitted_alias[1],
                                    )
                                    grapheme_tier.add(
                                        (otos[i + 1].offset + otos[i + 1].preutterance)
                                        / 1000,
                                        audio.duration_seconds,
                                        "SP",
                                    )
                                    phoneme_tier.add(
                                        (oto.offset + oto.preutterance) / 1000,
                                        (otos[i + 1].offset + otos[i + 1].preutterance)
                                        / 1000,
                                        phs[0],
                                    )
                                    phoneme_tier.add(
                                        (otos[i + 1].offset + otos[i + 1].preutterance)
                                        / 1000,
                                        audio.duration_seconds,
                                        "SP",
                                    )
                                elif len(phs) == 2:
                                    # phoneme = utaupy.label.Phoneme()
                                    # phoneme.symbol = phs[0]
                                    # phoneme.start = (oto.offset + oto.overlap) * 1000
                                    # phoneme.end = (oto.offset + oto.preutterance) * 1000
                                    # label.append(phoneme)
                                    # phoneme = utaupy.label.Phoneme()
                                    # phoneme.symbol = phs[1]
                                    # phoneme.start = (oto.offset + oto.preutterance) * 1000
                                    # phoneme.end = (otos[i + 1].offset + otos[i + 1].preutterance) * 1000
                                    # label.append(phoneme)
                                    # phoneme = utaupy.label.Phoneme()
                                    # phoneme.symbol = "SP"
                                    # phoneme.start = (otos[i + 1].offset + otos[i + 1].preutterance) * 1000
                                    # phoneme.end = audio_length
                                    # label.append(phoneme)
                                    grapheme_tier.add(
                                        (oto.offset + oto.overlap) / 1000,
                                        (otos[i + 1].offset + otos[i + 1].preutterance)
                                        / 1000,
                                        splitted_alias[1],
                                    )
                                    grapheme_tier.add(
                                        (otos[i + 1].offset + otos[i + 1].preutterance)
                                        / 1000,
                                        audio.duration_seconds,
                                        "SP",
                                    )
                                    phoneme_tier.add(
                                        (oto.offset + oto.overlap) / 1000,
                                        (oto.offset + oto.preutterance) / 1000,
                                        phs[0],
                                    )
                                    phoneme_tier.add(
                                        (oto.offset + oto.preutterance) / 1000,
                                        (otos[i + 1].offset + otos[i + 1].preutterance)
                                        / 1000,
                                        phs[1],
                                    )
                                    phoneme_tier.add(
                                        (otos[i + 1].offset + otos[i + 1].preutterance)
                                        / 1000,
                                        audio.duration_seconds,
                                        "SP",
                                    )
                                else:
                                    raise ValueError("Invalid phoneme length.")
                            elif len(next_phs) == 1:
                                if len(phs) == 1:
                                    # phoneme = utaupy.label.Phoneme()
                                    # phoneme.symbol = phs[0]
                                    # phoneme.start = (oto.offset + oto.preutterance) * 1000
                                    # phoneme.end = (otos[i + 1].offset + otos[i + 1].preutterance) * 1000
                                    # label.append(phoneme)
                                    grapheme_tier.add(
                                        (oto.offset + oto.preutterance) / 1000,
                                        (otos[i + 1].offset + otos[i + 1].preutterance)
                                        / 1000,
                                        splitted_alias[1],
                                    )
                                    phoneme_tier.add(
                                        (oto.offset + oto.preutterance) / 1000,
                                        (otos[i + 1].offset + otos[i + 1].preutterance)
                                        / 1000,
                                        phs[0],
                                    )
                                elif len(phs) == 2:
                                    # phoneme = utaupy.label.Phoneme()
                                    # phoneme.symbol = phs[0]
                                    # phoneme.start = (oto.offset + oto.overlap) * 1000
                                    # phoneme.end = (oto.offset + oto.preutterance) * 1000
                                    # label.append(phoneme)
                                    # phoneme = utaupy.label.Phoneme()
                                    # phoneme.symbol = phs[1]
                                    # phoneme.start = (oto.offset + oto.preutterance) * 1000
                                    # phoneme.end = (otos[i + 1].offset + otos[i + 1].preutterance) * 1000
                                    # label.append(phoneme)
                                    grapheme_tier.add(
                                        (oto.offset + oto.overlap) / 1000,
                                        (otos[i + 1].offset + otos[i + 1].preutterance)
                                        / 1000,
                                        splitted_alias[1],
                                    )
                                    phoneme_tier.add(
                                        (oto.offset + oto.overlap) / 1000,
                                        (oto.offset + oto.preutterance) / 1000,
                                        phs[0],
                                    )
                                    phoneme_tier.add(
                                        (oto.offset + oto.preutterance) / 1000,
                                        (otos[i + 1].offset + otos[i + 1].preutterance)
                                        / 1000,
                                        phs[1],
                                    )
                                else:
                                    raise ValueError("Invalid phoneme length.")
                            elif len(next_phs) == 2:
                                if len(phs) == 1:
                                    # phoneme = utaupy.label.Phoneme()
                                    # phoneme.symbol = phs[0]
                                    # phoneme.start = (oto.offset + oto.preutterance) * 1000
                                    # phoneme.end = (otos[i + 1].offset + otos[i + 1].overlap) * 1000
                                    # label.append(phoneme)
                                    grapheme_tier.add(
                                        (oto.offset + oto.preutterance) / 1000,
                                        (otos[i + 1].offset + otos[i + 1].overlap)
                                        / 1000,
                                        splitted_alias[1],
                                    )
                                    phoneme_tier.add(
                                        (oto.offset + oto.preutterance) / 1000,
                                        (otos[i + 1].offset + otos[i + 1].overlap)
                                        / 1000,
                                        phs[0],
                                    )
                                elif len(phs) == 2:
                                    # phoneme = utaupy.label.Phoneme()
                                    # phoneme.symbol = phs[0]
                                    # phoneme.start = (oto.offset + oto.overlap) * 1000
                                    # phoneme.end = (oto.offset + oto.preutterance) * 1000
                                    # label.append(phoneme)
                                    # phoneme = utaupy.label.Phoneme()
                                    # phoneme.symbol = phs[1]
                                    # phoneme.start = (oto.offset + oto.preutterance) * 1000
                                    # phoneme.end = (otos[i + 1].offset + otos[i + 1].overlap) * 1000
                                    # label.append(phoneme)
                                    grapheme_tier.add(
                                        (oto.offset + oto.overlap) / 1000,
                                        (otos[i + 1].offset + otos[i + 1].overlap)
                                        / 1000,
                                        splitted_alias[1],
                                    )
                                    phoneme_tier.add(
                                        (oto.offset + oto.overlap) / 1000,
                                        (oto.offset + oto.preutterance) / 1000,
                                        phs[0],
                                    )
                                    phoneme_tier.add(
                                        (oto.offset + oto.preutterance) / 1000,
                                        (otos[i + 1].offset + otos[i + 1].overlap)
                                        / 1000,
                                        phs[1],
                                    )
                                else:
                                    raise ValueError("Invalid phoneme length.")
                            else:
                                raise ValueError("Invalid phoneme length.")
                    tg.append(grapheme_tier)
                    tg.append(phoneme_tier)
                    tg.write(str(textgrid_dir / f"{wav_file.stem}.TextGrid"))
                    pbar.update(1)

            print()
            print("Phase 3: Done.")
            print()
        else:
            print("Invalid forced aligner.")
            sys.exit(1)
        print("Phase 4: Build dataset...")
        print()

        ctx = Context(build_dataset)
        with ctx:
            build_dataset.parse_args(
                ctx,
                [
                    "--wavs",
                    str(temp_dir),
                    "--tg",
                    str(temp_dir / "TextGrid"),
                    "--dataset",
                    str(temp_dir / "Dataset"),
                ],
            )
            build_dataset.invoke(ctx)

        outputs_path = pathlib.Path("src/outputs")
        output_path = outputs_path / datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        output_path.mkdir()
        shutil.move(temp_dir / "Dataset" / "transcriptions.csv", output_path)
        shutil.move(temp_dir / "Dataset" / "wavs", output_path)
        output_wavs_path = output_path / "wavs"

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
