## ICT-NLP System for SemEval-2026 Task 3 (DimABSA) – Track A, Subtask 1

This repository contains the official implementation of team **ICT-NLP** for **SemEval-2026 Task 3: Dimensional Aspect-Based Sentiment Analysis (DimABSA)**, specifically **Track A – Subtask 1 (DimASR)**.  
The task and datasets are described in the shared task repository: [DimABSA/DimABSA2026](https://github.com/DimABSA/DimABSA2026).

Our system is a **multilingual transformer-based regressor** that predicts **valence–arousal (VA)** scores for given aspects across multiple languages and domains.

---

### Task Description (Track A – DimASR)

In **Dimensional Aspect Sentiment Regression (DimASR)**, each instance consists of:

- **Text**: a sentence or paragraph expressing opinions.
- **Aspect(s)**: one or more aspect terms mentioned in the text.

The goal is to predict a **real-valued VA pair** \((V, A)\) for each aspect, where:

- \(V\) (valence) and \(A\) (arousal) are in the range **\[1.00, 9.00\]**.
- Scores are evaluated with **RMSE** as defined in the official evaluation script.

This repository focuses on **Track A, Subtask 1** for the following **language–domain combinations**:

- `eng`: `laptop`, `restaurant`
- `zho`: `restaurant`, `laptop`, `finance`
- `jpn`: `hotel`, `finance`
- `rus`: `restaurant`
- `tat`: `restaurant`
- `ukr`: `restaurant`

---

### System Overview

- **Backbone**: HuggingFace **transformer encoder** (default: `bert-base-multilingual-cased`, configurable to other models such as XLM-R).
- **Input representation**:
  - Uses the **aspect** and the **full sentence** as **two segments**:
    - Segment A: aspect text.
    - Segment B: sentence text.
  - Implemented via the tokenizer’s two-text interface:
    - `[CLS] aspect [SEP] text [SEP]`, with `token_type_ids` when supported.
- **Regression head**:
  - Uses the `[CLS]` representation.
  - Applies a dropout layer followed by a linear layer to output **2 dimensions**: \([V, A]\).
  - A `sigmoid` + linear scaling restricts outputs to **\[1, 9\]** for both V and A.
- **Loss and optimization**:
  - Objective: **MSE loss** between predicted and gold VA.
  - Optimizer: **AdamW**.
  - **Early stopping** on dev **RMSE\_VA** with patience 3.
- **Training regime**:
  - **All languages and domains are merged** into a single training pool by default.
  - A **90/10 train–dev split** is performed on the combined data (with fixed random seed for reproducibility).
- **Prediction and ensembling**:
  - For each language–domain combination, the trained model outputs VA predictions in the **official JSONL format**.
  - An optional **ensemble module** averages predictions from multiple checkpoints per language–domain combination.

---

### Repository Structure

- **Root**
  - `README.md`: This file.
- **`subtask_1/`**
  - `main.py`: Entry point for training and prediction (DimASR, Track A, Subtask 1).
  - `train.py`: Training loop on the combined multilingual, multidomain dataset.
  - `predict.py`: Generates predictions for dev/test sets and writes JSONL outputs.
  - `model.py`: Defines the `TransformerVARegressor` model (transformer backbone + VA regression head).
  - `Dataset.py`: Defines `VADataset` (aspect–sentence pair tokenization and label construction).
  - `utils.py`:
    - Reproducibility utilities (`set_global_seed`, `worker_init_fn`).
    - JSONL loading/conversion helpers (`load_jsonl`, `jsonl_to_df`, `df_to_jsonl`).
    - `Tee` class to mirror logs to both console and file.
  - `evaluation.py`: Utility functions for Dev-time evaluation:
    - `get_prd`: batches forward passes and collects predictions.
    - `evaluate_predictions_task1`: computes **Pearson correlations** and **RMSE\_VA**.
  - `example_data_processing.py`: Demonstration script showing how a single example flows through `jsonl_to_df` and `VADataset`.
  - `config.yaml`: Configuration file for Track A, Subtask 1.
---

### Requirements

- **Python** ≥ 3.9
- **PyTorch**
- **Transformers** (HuggingFace)
- **pandas**
- **numpy**
- **scikit-learn**
- **scipy**
- **PyYAML**
- **tqdm**

> Note: The code uses `os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'` to support mirrored HuggingFace endpoints. You may remove or adapt this line depending on your environment.

---

### Citation

If you use this code or ideas from our system in your research, please cite the DimABSA shared task paper and dataset paper (from the official repository):

```bibtex
@inproceedings{yu-etal-2026-semeval,
  title        = {{S}em{E}val-2026 Task 3: Dimensional Aspect-Based Sentiment Analysis ({D}im{ABSA})},
  author       = {Yu, Liang-Chih and Becker, Jonas and Muhammad, Shamsuddeen Hassan and Abdulmumin, Idris and Lee, Lung-Hao and Lin, Ying-Lung and Wang, Jin and Wahle, Jan Philip and Ruas, Terry and Panchenko, Alexander and Alimova, Ilseyar and Chang, Kai-Wei and Wanzare, Lilian and Odhiambo, Nelson and Gipp, Bela and Mohammad, Saif M.},
  year         = 2026,
  booktitle    = {Proceedings of the 20th International Workshop on Semantic Evaluation (SemEval-2026)},
  publisher    = {Association for Computational Linguistics}
}
```

```bibtex
@misc{lee2026dimabsabuildingmultilingualmultidomain,
      title={DimABSA: Building Multilingual and Multidomain Datasets for Dimensional Aspect-Based Sentiment Analysis}, 
      author={Lung-Hao Lee and Liang-Chih Yu and Natalia Loukashevich and Ilseyar Alimova and Alexander Panchenko and Tzu-Mi Lin and Zhe-Yu Xu and Jian-Yu Zhou and Guangmin Zheng and Jin Wang and Sharanya Awasthi and Jonas Becker and Jan Philip Wahle and Terry Ruas and Shamsuddeen Hassan Muhammad and Saif M. Mohammad},
      year={2026},
      eprint={2601.23022},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2601.23022}, 
}
```

Please also cite the **ICT-NLP system description paper**:

```bibtex
@inproceedings{huang-etal-2026-ict,
    title = "{ICT}-{NLP} at {S}em{E}val-2026 Task 3: Less Is More {---} Multilingual Encoder with Joint Training and Adaptive Ensemble for Dimensional Aspect Sentiment Regression",
    author = "Huang, Liyuan  and
      He, Jiawei  and
      Shen, Wutao  and
      Li, Lin  and
      Zhang, Jin",
    booktitle = "Proceedings of the 20th {I}nternational {W}orkshop on {S}emantic {E}valuation (2026)",
    month = jul,
    year = "2026",
    address = "San Diego, California, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.semeval-1.131/",
    doi = "10.18653/v1/2026.semeval-1.131",
    pages = "950--957",
    ISBN = "979-8-89176-414-9"
}
```
