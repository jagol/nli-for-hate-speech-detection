# Hypothesis Engineering for Zero-Shot and Few-Shot Hate Speech Detection

This code accompanies the paper: Hypothesis Engineering for Zero-Shot and Few-Shot Hate Speech Detection.

## Setup

Collect the HateCheck and ETHOS datasets:
```bash
mkdir data data/HateCheck data/ETHOS_Binary
wget -O data/HateCheck/HateCheck_test.csv https://raw.githubusercontent.com/paul-rottger/hatecheck-data/main/test_suite_cases.csv
wget -O data/ETHOS_Binary/Ethos_Dataset_Binary.csv https://raw.githubusercontent.com/intelligence-csd-auth-gr/Ethos-Hate-Speech-Dataset/master/ethos/ethos_data/Ethos_Dataset_Binary.csv
```

Create a python environment and install the required packages:
```bash
python3 -m venv nli_for_hs_venv
source nli_for_hs_venv/bin/activate
pip install -r requirements.txt
```

Create a file `paths.json` in the repository's root directory and write to it:
```json
{
  "data_dir": "path/to/the/data/directory",
  "output_dir": "path/to/the/output/directory",
  "configs_dir": "path/to/the/configs/directory"
}
```
Checkpoints, logs and results will be written to the output directory.

As an example:
- the `data_dir` could be `/home/user/projects/nli-for-hate-speech-detection/data/` (in the following sections referenced as `<path-to-data-dir>`)
- `output_dir` could be `/home/user/projects/nli-for-hate-speech-detection/output/` (in the following sections referenced as `<path-to-output-dir>`)
- and `configs_dir` could be `configs/` (in the following sections referenced as `<path-to-configs-dir>`).

## Run Zero-Shot Experiments

To compare the performance of various hypotheses on HateCheck run:
```bash
./compare_hypotheses_on_HateCheck.sh <path-to-configs-dir> <gpu-num>
```

To compare the performance of the proposed strategies on HateCheck run:
```bash
./compare_strategies_on_HateCheck.sh <path-to-configs-dir> <gpu-num>
```

To compare the performance of the proposed strategies on ETHOS run:
```bash
./compare_strategies_on_ETHOS.sh <path-to-configs-dir> <path-to-data-dir> <gpu-num>
```

## Experiment With Your Own Hypotheses and Strategies

Instruction to implement your own strategies coming soon...
