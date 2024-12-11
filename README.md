# Task Guided Multimodal Object Search for Service Robots via Reinforcement Learning


## Setup

1. (Recommended) Create a virtual environment using virtualenv or conda:

```
conda create -n TGMOS python=3.6
conda activate TGMOS
```

2. Clone the repository as:
```
git clone https://github.com/Li-XD-Pro/Task-Guided-Object-Search.git
cd Task-Guided-Object-Search
```

3. For the rest of dependencies, please run 
```
pip install -r requirements.txt --ignore-installed
```


## Data

The offline data can be found [here](https://drive.google.com/drive/folders/1i6V_t6TqaTpUdUFpOJT3y3KraJjak-sa?usp=sharing).

"data.zip" (~5 GB) contains everything needed for evalution. Please unzip it and put it into the Zero-Shot-Object-Navigation folder.

For training, please also download "train.zip" (~9 GB), and put all "Floorplan" folders into `./data/thor_v1_offline_data`

## Evaluation

Note: if you are not using gpu, you can remove the argument `--gpu-ids 0`

Evaluate our model under 18/4 class split

```bash
python main.py --eval \
    --test_or_val test \
    --episode_type TestValEpisode \
    --load_model pretrained_models/yourmodelname.dat \
    --model MJOLNIR_R \
    --gpu-ids 0 \
    --zsd 1 \
    --split 18/4
```

Evaluate our model under 14/8 class split

```bash
python main.py --eval \
    --test_or_val test \
    --episode_type TestValEpisode \
    --load_model pretrained_models/yourmodelname.dat \
    --model MJOLNIR_R \
    --gpu-ids 0 \
    --zsd 1 \
    --split 14/8
```

## Training

Note: the folder to save trained model should be set up before training.

Train our model under 18/4 class split

```bash
python main.py \
    --title mjolnir_train \
    --model MJOLNIR_R \
    --gpu-ids 0 \
    --workers 8 \
    --vis False \
    --save-model-dir trained_models/MR_18_4/ \
    --zsd 1 \
    --partial_reward 1 \
    --split 18/4
```

Train our model under 14/8 class split

```bash
python main.py \
    --title mjolnir_train \
    --model MJOLNIR_R \
    --gpu-ids 0 \
    --workers 8 \
    --vis False \
    --save-model-dir trained_models/MR_14_8/ \
    --zsd 1 \
    --partial_reward 1 \
    --split 14/8
```