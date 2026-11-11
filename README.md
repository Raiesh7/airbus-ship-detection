# Airbus Ship Detection

Image-level classification of Airbus satellite images as **Ship** or **No Ship**. The single source of code is a Kaggle-ready notebook that compares five traditional classifiers using HSV-histogram + HOG features with a compact CNN. The project predicts whether an image contains at least one ship; it does not perform ship localization or segmentation.

## Assignment deliverables

| Requirement | Repository item |
|---|---|
| GitHub source code and setup instructions | `notebooks/Airbus_Ship_Detection_Kaggle.ipynb` and this README |
| Concise PDF write-up | `docs/Airbus_Ship_Detection_Report.pdf` |
| Presentation for review | `docs/Airbus_Ship_Detection_Presentation.pptx` |
| Results and visual evidence | `outputs/metrics.json`, `outputs/model_comparison.csv`, and `outputs/plots/` |

## Repository layout

```text
notebooks/Airbus_Ship_Detection_Kaggle.ipynb  Complete executable Kaggle workflow
data/README.md                               Dataset attachment instructions
docs/Airbus_Ship_Detection_Report.pdf        Project write-up
docs/Airbus_Ship_Detection_Presentation.pptx Presentation deck
outputs/                                      Recorded metrics and plots
requirements.txt                              Local dependency reference
```

## Run directly on Kaggle

1. Create a Kaggle Notebook and upload/import `notebooks/Airbus_Ship_Detection_Kaggle.ipynb`.
2. Click **Add Input** and attach the original **Airbus Ship Detection Challenge** dataset.
3. Accept the competition rules if Kaggle asks.
4. Select a GPU accelerator if available; the CNN also runs on CPU but is slower.
5. Run every cell from top to bottom.

The notebook automatically searches `/kaggle/input` for `train_ship_segmentations_v2.csv` and either `train_v2/` or `train_v2.zip`. It writes metrics, models, confusion matrices, and a downloadable ZIP to `/kaggle/working/airbus_project_outputs/`.

The default run uses a stratified 3,000-image subset for traditional ML and a 5,000-image subset for the CNN so it is practical in a Kaggle session. Change `TRADITIONAL_LIMIT` or `CNN_LIMIT` in the configuration cell for a larger experiment. Do not commit the dataset or trained models.

## Recorded baseline

The committed CSV/JSON are recovered baseline artifacts from a prior notebook run and must be regenerated for the final demonstration. They use separate stratified held-out splits: 600 traditional-ML test images and 1,000 CNN test images.

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Random Forest | 0.857 | 0.851 | 0.429 | 0.570 |
| CNN | 0.787 | 0.563 | 0.163 | 0.253 |
| Gaussian Naive Bayes | 0.477 | 0.286 | 0.910 | 0.435 |
| Linear SVM | 0.713 | 0.367 | 0.406 | 0.386 |
| LDA | 0.730 | 0.347 | 0.248 | 0.289 |
| KNN | 0.790 | 0.889 | 0.060 | 0.113 |

Because the dataset is imbalanced, precision, recall, F1-score, and confusion matrices should be discussed alongside accuracy.

## Local setup

The project is designed for Kaggle execution. If running locally, install the packages in `requirements.txt`, place the CSV and image archive/folder according to [data/README.md](data/README.md), and adapt the notebook's `INPUT_ROOT` to that local path.
