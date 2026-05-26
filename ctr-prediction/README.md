# AuGR for CTR Prediction

This folder contains code used to produce experimental results for AuGR on CTR prediction dataset (Taobao Ads).

## Setup and data preparation

### 1) Download datasets

The CTR data zips are hosted in the reczoo/datasets repository:

- https://github.com/reczoo/datasets

Download the TaobaoAd zip files from that repo and extract them locally.

### 2) Prepare the dataset folders

Each dataset folder should contain the CSV files and the model config JSON used by the
training scripts, for example:

```
TaobaoAd_x1/
	train.csv
	valid.csv
	test.csv
	taobao_grouping_model_config_v1.json
```

### 3) Generate vocab dictionaries (optional)

The training scripts expect a vocab JSON that you generate from the training CSV. The repository includes vocab dicts pregenerated, but you may use the following scripts to generate your own:

```
python -m src.indexing.gen_taobao_vocab \
	--input_path /path/to/TaobaoAd_x1/train.csv \
	--output_path /path/to/TaobaoAd_x1/taobao_vocab.json
```

### 4) Run training with the provided scripts

The training launchers live in [src/training/scripts](src/training/scripts). They are
intended to be run from the repo root and will `cd` internally as needed.

#### Taobao

```
bash src/training/scripts/run_train_taobao_ctr_v1.sh <exp_name>
```

Update these fields for your environment in
[src/training/scripts/run_train_taobao_ctr_v1.sh](src/training/scripts/run_train_taobao_ctr_v1.sh#L11-L95):

- `DATASET_ROOT` (or pass it as the second argument).
- `TRAIN_DATA_PATH`, `VALID_DATA_PATH`, `TEST_DATA_PATH` if your CSV names differ. Note that TaoBao dataset does not have its own validation data split, the base script provided uses test.csv for both VALID and TEST splits.
- `TAOBAO_VOCAB_PATH` (the vocab JSON generated in step 3).
- `MODEL_CONFIG_PATH` (the grouping config JSON located in the dataset folder).
- Optional: feature lists (`CATEGORICAL_FEATURES`, `ITEM_FEATURES`, `SEQUENCE_FEATURES`)
	if your CSV columns differ from the defaults.
- Optional: `WANDB_API_KEY` / `WANDB_PROJECT` / `WANDB_RUN_NAME` if you use W&B.