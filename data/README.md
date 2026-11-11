# Dataset

The dataset is intentionally not stored in this repository. In Kaggle Notebooks, use **Add Input** to attach the original Airbus Ship Detection Challenge dataset and accept its competition terms if prompted. The notebook searches recursively under `/kaggle/input` for `train_ship_segmentations_v2.csv` and either the `train_v2/` image directory or `train_v2.zip`.

For local execution, place the annotation CSV and image folder here:

```text
data/
├── train_ship_segmentations_v2.csv
└── train_v2/
    └── <image files>.jpg
```

Do not commit the CSV, image archive, or extracted image files.
