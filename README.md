# Voicebank2DiffSinger
UTAUの音源ファイルからSOFAとMakeDiffSingerを用いて、学習前のデータセットを生成する

## 前提要件
- C++ によるデスクトップ開発 (Visual Studio)
- CMake

## 使い方 (Windows)
1. このリポジトリをsubmoduleを含めcloneする
    ```sh
    git clone --recursive
    ```
2. 仮想環境を構築し、入る
    ```sh
    python -m venv .venv
    .venv/scripts/activate
    ```
3. 必要なモジュールをインストールする
    ```sh
    pip install -r requirements.txt
    pip install -r src/SOFA/requirements.txt
    pip install -r src/MakeDiffSinger/acoustic_forced_alignment/requirements.txt
    pip install -r src/MakeDiffSinger/variance-temp-solution/requirements.txt
    ```
4. [PyTorchの公式サイト](https://pytorch.org/get-started/locally/)にて、セットアップをする
5. [日本語のSOFAモデル](https://github.com/colstone/SOFA_Models/releases/tag/JPN-V0.0.2b)をダウンロードし、解凍後中にある「japanese-v2.0-45000.ckpt」を「src/cktp」に配置する
6. src/main.py の args に音源フォルダを一つ(もしくは複数)渡し起動する
    ```sh
    python src/main.py example/A3 example/A2 example/A4
    ```
