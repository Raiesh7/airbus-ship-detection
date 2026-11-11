# Airbus Ship Detection: Traditional ML and CNN

## Problem statement

This project asks a simple question: does a satellite image contain at least one ship? Images from the Airbus Ship Detection Challenge are classified as **Ship** or **No Ship**. The system works at image level; it does not draw bounding boxes or produce ship masks.

## Dataset

The Kaggle dataset contains satellite images and run-length encoded annotations. The annotation table is reduced to one binary label per image: an image is marked Ship when at least one annotation is present. The attached run contained 192,556 labelled images. The classes are imbalanced, with 42,556 ship images and 150,000 no-ship images. To keep the notebook practical, the experiment used stratified subsets rather than processing every image.

## Approach and implementation

The complete implementation is in `notebooks/Airbus_Ship_Detection_Kaggle.ipynb`. It finds the Kaggle input automatically, works with either an extracted `train_v2/` directory or `train_v2.zip`, resizes images to 128x128, and displays labelled examples before training.

For the traditional models, each image is represented by a normalized HSV colour histogram joined with HOG texture and shape features. Random Forest, Linear SVM, KNN, Gaussian Naive Bayes, and Linear Discriminant Analysis are evaluated using a stratified 80/20 split. Accuracy, precision, recall, F1-score, and a confusion matrix are recorded for each model.

The CNN uses a streaming Keras sequence, loading images in batches instead of placing the whole sample in memory. It has three convolution/max-pooling blocks, global average pooling, dropout, and dense layers. Class weights are used to reduce the effect of the class imbalance, and early stopping monitors validation loss. Metrics, plots, trained models, and a ZIP archive are saved under `/kaggle/working/airbus_project_outputs/`.

## Baseline results

The following results come from the completed Kaggle run included with this submission. Traditional models used 3,000 images, while the CNN used a stratified sample of 5,000 images.

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Random Forest | 0.857 | 0.851 | 0.429 | 0.570 |
| CNN | 0.787 | 0.563 | 0.163 | 0.253 |
| Gaussian Naive Bayes | 0.477 | 0.286 | 0.910 | 0.435 |
| Linear SVM | 0.713 | 0.367 | 0.406 | 0.386 |
| LDA | 0.730 | 0.347 | 0.248 | 0.289 |
| KNN | 0.790 | 0.889 | 0.060 | 0.113 |

Random Forest is the strongest model in this run, with an F1-score of 0.570 and precision of 0.851. The CNN reaches 0.787 accuracy, but its recall is only 0.163, so it misses many ship images at the default decision threshold. This contrast shows why accuracy alone is not enough for this imbalanced dataset. The confusion matrices and precision/recall values give a more useful view of the result.

## Conclusion

The experiment provides a reproducible Kaggle comparison between handcrafted image features and a CNN. The best current baseline is the Random Forest model. The main limitations are the image-level target, the reduced training samples, and the CNN's low ship recall. Further work should try transfer learning, augmentation, decision-threshold tuning, and the original segmentation annotations for ship localization.
