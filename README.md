# Attack-PTMC

This is the codebase for the paper "An Extensive Study on Adversarial Attack against Pre-trained Models of Code".

## Attack Approach

- [MHM](https://github.com/SEKE-Adversary/MHM)
- [ACCENT](https://github.com/zhangxq-1/ACCENT-repository)
- [ALERT](https://github.com/soarsmu/attack-pretrain-models-of-code/)
- [WIR-Random](https://github.com/ZZR0/ISSTA22-CodeStudy)
- [StyleTransfer](https://github.com/mdrafiqulrabin/JavaTransformer)

## Experiments


**Create Environment**

```
pip install -r requirements.txt
```

**Model Fine-tuning**

Use `train.py` to train models. We also provide our pre-trained models [here](https://zenodo.org/record/7613725#.Y-G3SNpBxPY).

Take an example:

```
cd CodeBERT/Clone-detection/code
python train.py
```

**Running Existing Attacks**

In our study, we employed six distinct datasets: BigCloneBench and Google Code Jam for clone detection, OWASP and Juliet Test Suite for vulnerability detection, and CodeSearchNet and TLCodeSum for code summarization. Among them, BigCloneBench, OWASP and CodeSearchNet are used for pre-study, and all six datasets are used for the evaluation of our approach.

For pre-study, you should download the Dataset and Model from [Zenodo](https://zenodo.org/record/7613725#.Y-G3SNpBxPY) and place the file in the appropriate path. In Zenodo the Dataset and Model are placed in the directory that corresponds to the code.

Take an example:

```
cd CodeBERT/Clone-detection/attack
python run_xxx.py
```

The `run_xxx.py` here can be `run_mhm.py`, `run_alert.py`, `run_style.py`, `run_wir_random.py`, `run_beam.py`

Take `run_mhm.py`  as an example:

```
import os

os.system("CUDA_VISIBLE_DEVICES=1 python attack_mhm.py \
    --output_dir=../saved_models \
    --model_type=roberta \
    --tokenizer_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --csv_store_path result/attack_mhm_all.csv \
    --original\
    --base_model=microsoft/codebert-base-mlm \
    --eval_data_file=../../../dataset/Clone-detection/test_sampled.txt \
    --block_size 512 \
    --eval_batch_size 2 \
    --seed 123456  2>&1")
```

And the result will saved in `./result/attack_mhm_all.csv`.

Run experiments on other tasks of other models as well.

`./CodeBERT/` contains code for the CodeBERT experiment and `./CodeGPT` contains code for CodeGPT experiment and  `./PLBART ` contains code for PLBART experiment.

**Running ACCENT**

The attack approach ACCENT can refer to [ACCENT](https://github.com/zhangxq-1/ACCENT-repository)'s experimental run description.

> Note: in the path ACCENT/Clone-detection and ACCENT/Vulnerability-detection, the directory `evaluation/`, `c2nl` and `python_parser`  are the same as directory of the same name under ACCENT/Code-summarization.

**Evaluation Our Approach**

For the evaluation of our approach, you should run the `attack_beam.py`. The code and results pertaining to the datasets BigCloneBench, OWASP, and CodeSearchNet are available in [Zenodo](https://zenodo.org/record/7613725#.Y-G3SNpBxPY). For model training and the implementation of attacks on the other three datasets is consistent with the code corresponding to the above datasets. One simply needs to modify the line of code loading the data, as seen in the parameter `eval_data_file` in the above `run_xxx.py`. We have placed these three datasets, along with their attack results, under the directory `Dataset and Result/`.

## Target Models and Datasets

### Models

The pre-trained models can be downloaded from this [Zenodo](https://zenodo.org/record/7613725#.Y-G3SNpBxPY). After decompressing this file, the folder structure is as follows.

```
CodeBERT/
|-- Clone-detection
|   `-- saved_models
|       |-- checkpoint-best-f1
|       |   `-- model.bin
|       |-- test.log
|       `-- train.log
|-- Code-summarization
|   `-- saved_models
|       |-- checkpoint-best-bleu
|       |   `-- pytorch_model.bin
|       |-- checkpoint-best-ppl
|       |   `-- pytorch_model.bin
|       |-- checkpoint-last
|       |   `-- pytorch_model.bin
|       |-- test.log
|       `-- train.log
`-- Vulnerability-detection
    `-- saved_models
        |-- checkpoint-best-acc
        |   `-- model.bin
        |-- test.log
        `-- train.log
CodeGPT/
...
PLBART/
...
```

### Datasets and Results

The datasets and results can be downloaded from this [Zenodo](https://zenodo.org/record/7613725#.Y-G3SNpBxPY). After decompressing this file, the folder structure is as follows.

```
CodeBERT/
|-- Clone-detection
|   `-- result
|       |-- attack_accent_all.csv
|       |-- attack_alert_all.csv
|       |-- attack_beam_all.csv
|       |-- attack_mhm_all.csv
|       |-- attack_style_all.csv
|       `-- attack_wir_all.csv
|-- Code-summarization
|   `-- result
|       |-- attack_accent_all.csv
|       |-- attack_alert_all.csv
|       |-- attack_beam_all.csv
|       |-- attack_mhm_all.csv
|       |-- attack_style_all.csv
|       `-- attack_wir_all.csv
`-- Vulnerability-detection
    `-- result
        |-- attack_accent_all.csv
        |-- attack_alert_all.csv
        |-- attack_beam_all.csv
        |-- attack_mhm_all.csv
        |-- attack_style_all.csv
        `-- attack_wir_all.csv
CodeGPT/
...
PLBART/
...
```

Take `attack_mhm_all.csv` as an example

The csv file contains 9 columns

- `Index: `The number of each sample, 0-3999.
- `Original Code: `The original code before the attack.
- `Adversarial Code: `The adversarial sample code obtained after the successful attack.
- `Program Length: `The length of the code in code token.
- `Identifier Num: `The number of identifiers that the code can extract to.
- `Replaced Identifiers: `Information about identifier replacement in case of a successful attack.
- `Query Times: `Number of query model per attack.
- `Time Cost: `The time consumed per attack, in minutes.
- `Type: ` `0` if it fails, and the method of attack if it succeeds, e.g. `MHM`, `ALERT`, `ACCENT`, ... .

### Evaluate Result

Run the python script `eval.py` to get the analysis of csv based on csv.

```
cd evaluation
python eval.py
```

In `eval.py` you should change the csv path. And the `eval.py` can evaluate the `attack_accent_all.csv`, `attack_alert_all.csv`, `attack_beam_all.csv`, `attack_mhm_all.csv`, `attack_wir_all.csv`. 

Only `attack_style_all.csv` you should run the `eval_style.py`.

The difference between `eval.py` and `eval_style.py`  is that the latter does not need to calculate ICR, TCR, ACS, AED metrics. 

## Acknowledgement

We are very grateful that the authors of CodeBERT, CodeGPT, PLBART, MHM , ALERT, ACCENT, WIR-Random, StyleTransfer make their code publicly available so that we can build this repository on top of their code. 
