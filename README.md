# Enhancing Machine Learning Models for Medical Coding: Diagnostic Codes Mapping and Synthetic Clinical Notes

Discharge summaries are free-text notes that outline the hospital stay of a patient, including diagnoses, treatments, and follow-up care recommendations. Through the use of International Classification of Diseases (ICD) coding, they play a vital role in communicating clinical data exchange and reimbursement. However, manually assigning ICD codes to discharge summaries can be labor-intensive and prone to errors due to the unstructured narrative format, variations in terminology, potential inaccuracies in documentation, and the limitations of the ICD coding system in capturing the complexity of patient conditions. This repository implements state-of-the-art automated medical coding machine learning models. A data management method is utilized to maximize the amount of usable data by mapping ICD-9 codes in the MIMIC-IV dataset to their corresponding ICD-10. The results of this extensive preprocessing are evaluated with the use of several well-established and state-of-the-art deep learning models, which present significant improvement.

### Features of our repository

* **Model Execution Pipeline**
  Complete data splits, preprocessing scripts, and training/evaluation code for six canonical models, allowing others to verify and compare results.

* **HuggingFace Dataset Wrappers**
  Ready-to-use `dataset` loader for MIMIC-IV.

* **ICD-9 → ICD-10 Mapping Modules**  
  Integration of [simple_icd_10](https://github.com/StefanoTrv/simple_icd_10), [simple_icd_10_CM](https://github.com/StefanoTrv/simple_icd_10_CM), and [ICD-Mappings](https://github.com/snovaisg/ICD-Mappings) for robust, community-driven conversion of ICD-9 codes to ICD-10 in the MIMIC-IV dataset.

* **High-Volume Data Optimization**  
  Source-code optimizations throughout the preprocessing and training pipelines to efficiently handle larger cohorts and higher throughput of clinical notes.

## Supported Models

| Model       | Source Study                                                                                     |
| ----------- | ------------------------------------------------------------------------------------------------ |
| [CNN](https://github.com/jamesmullenbach/caml-mimic) | Explainable Prediction of Medical Codes from Clinical Text [NAACL 2018](https://aclanthology.org/N18-1100/)     |
| [Bi-GRU](https://github.com/jamesmullenbach/caml-mimic) | Mullenbach *et al.* [NAACL 2018](https://aclanthology.org/N18-1100/)                                                            |
| [CAML](https://github.com/jamesmullenbach/caml-mimic) | Mullenbach *et al.* [NAACL 2018](https://aclanthology.org/N18-1100/)                                                              |
| [MultiResCNN](https://github.com/foxlf823/Multi-Filter-Residual-Convolutional-Neural-Network) | Li *et al.* “ICD Coding from Clinical Text Using Multi-Filter Residual CNN” [arXiv 2019](https://arxiv.org/pdf/1912.00862.pdf)         |
| [LAAT](https://github.com/aehrc/LAAT) | Xiao *et al.* “A Label Attention Model for ICD Coding from Clinical Text” [ACL 2020](https://arxiv.org/abs/2007.06351)             |
| [PLM-ICD](https://github.com/MiuLab/PLM-ICD) | Wang *et al.* “PLM-ICD: Automatic ICD Coding with Pretrained Language Models” [ClinicalNLP 2022](https://aclanthology.org/2022.clinicalnlp-1.2/) |



## Installation

1. **Create a Python 3.10 or higher environment**

   ```bash
   conda create -n medcode python=3.10
   conda activate medcode
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## Experiments Pipeline

### Data Preparation (MIMIC-IV v2.2)

1. Download MIMIC-IV and MIMIC-IV-NOTE:

   * [https://physionet.org/content/mimiciv/2.2/](https://physionet.org/content/mimiciv/2.2/)
   * [https://physionet.org/content/mimic-iv-note/2.2/](https://physionet.org/content/mimic-iv-note/2.2/)

2. Set paths in `src/settings.py`:

   ```python
   DOWNLOAD_DIRECTORY_MIMICIV = "/path/to/mimiciv"
   DOWNLOAD_DIRECTORY_MIMICIV_NOTE = "/path/to/mimiciv-note"
   ```
3. Data preprocessing complete with text and code cleaning and code mapping:

   ```bash
   python prepare_data/prepare_mimiciv_icd10.py
   ```

4. Generate the training/validation/test subset splits:

   ```bash
   python prepare_data/prepare_mimiciv_splits_icd10.py
   ```

### Training

Use the Hydra-based configs under `configs/experiments`.
If you want to train PLM-ICD, you need to download [RoBERTa-base-PM-M3-Voc](https://dl.fbaipublicfiles.com/biolm/RoBERTa-base-PM-M3-Voc-hf.tar.gz), unzip it and change the `model_path` parameter in `configs/model/plm_icd.yaml` and `configs/text_transform/huggingface.yaml` to the path of the download. 

Examples:
* **LAAT on MIMIC-IV ICD-10**

  ```bash
  python main.py experiment=mimiciv_icd10/laat gpu=1
  ```

* **PLM-ICD on MIMIC-IV ICD-10**

  ```bash
  python main.py experiment=mimiciv_icd10/plm_icd gpu=0 callbacks=no_wandb trainer.print_metrics=True
  ```

To disable Weights & Biases logging:

```bash
python main.py ... callbacks=no_wandb trainer.print_metrics=true
```



## Repository Layout

```
├── configs/        # Hydra configs for experiments
├── files/          # Pre-existing data and static assets
├── prepare_data/   # Scripts to build datasets & splits
├── src/            # Core training, modeling, and utils
└── tests/          # Unit tests for data & modeling pipelines
```

## Known Limitations

* **Training Instability**
  Both LAAT and original PLM-ICD may diverge due to softmax overflow; normalization fixes are under investigation.

* **Memory Footprint**
  MIMIC-IV pipelines may exceed 32 GB RAM; 128 GB was used in development.


## Contributing

1. Fork this repository.
2. Create a feature branch: `git checkout -b feature/my-change`
3. Commit your changes and push: `git push origin feature/my-change`
4. Open a pull request detailing your improvements.

Please ensure any additions come with appropriate tests under `tests/`.

## Citation

The paper will be published in the 2025 IEEE Conference on Computational Intelligence in Bioinformatics and Computational Biology (CIBCB). A complete citation of our work will be published after the presentation of our work.

## Acknowledgements

Funded by the European Union - NextGenerationEU through Greece 2.0—National Recovery and Resilience Plan, under the call ”Flagship actions in interdisciplinary scientific fields with a special focus on the productive fabric” (ID16618), project name ”Bridging big omic, genetic and medical data for Precision Medicine implementation in Greece” (project code TAEDR-0539180).

This work builds on the foundational efforts of:

* **Joakim Edin**, **Alexander Junge**, **Jakob D. Havtorn**, **Lasse Borgholt**, **Maria Maistro**, **Tuukka Ruotsalo**, and **Lars Maaløe** for the original SIGIR 2023 [reproducibility study](https://github.com/JoakimEdin/medical-coding-reproducibility/tree/main).
* **Sotiris Lamprinidis** for the multi-label stratification algorithm and preprocessing utilities.
* All authors of the underlying model implementations (CAML, MultiResCNN, LAAT, PLM-ICD).
