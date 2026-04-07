# Experiments Log

## Experiment 1: Baseline ResNet18

### Model
- Architecture: ResNet18 (pretrained)
- Final layer modified for 4 classes

### Data
- Train/Val/Test split: 70/15/15
- Dataset: TrashNet

### Training Config
- Epochs: 5
- Batch size: 32
- Optimizer: Adam
- Learning rate: 0.001

### Results
- Best Validation Accuracy: 77.44%
- Epoch of Best Accuracy: 3

### Observations
- Model converges quickly
- Slight overfitting after epoch 3
- Validation accuracy stabilizes

### Test Results

- Test Accuracy: 76.74%

#### Confusion Matrix

```
| Actual \ Predicted | glass | metal | paper | plastic |
|------------------- |-------|-------|-------|---------|
| glass              | 45    | 21    | 1     | 9       |
| metal              | 3     | 56    | 2     | 1       |
| paper              | 2     | 2     | 79    | 7       |
| plastic            | 5     | 11    | 6     | 51      |
```

#### Classification Report

- Glass:
  - Precision: 0.82
  - Recall: 0.59
- Metal:
  - Precision: 0.62
  - Recall: 0.90
- Paper:
  - Precision: 0.90
  - Recall: 0.88
- Plastic:
  - Precision: 0.75
  - Recall: 0.70

### Detailed Observations

- The model performs best on **paper** with high precision and recall (~0.90), indicating strong feature learning for this class.
- **Metal** shows high recall (0.90) but lower precision (0.62), suggesting the model correctly identifies most metal items but also misclassifies other classes as metal.
- **Glass** has relatively low recall (0.59), indicating the model struggles to correctly identify all glass samples.
- Confusion is observed between:
  - Glass and plastic
  - Plastic and metal
- Overall test accuracy (76.74%) is slightly lower than validation accuracy (77.44%), indicating good generalization with minimal overfitting.

### Conclusion

The baseline ResNet18 model achieves a test accuracy of 76.74%. While performance is strong for certain classes like paper, there is room for improvement in distinguishing visually similar classes such as glass and plastic. Future improvements will focus on enhanced data augmentation and fine-tuning deeper layers of the network.

### Next Steps
- Add data augmentation
- Fine-tune deeper layers
- Increase epochs

## Experiment 2: ResNet18 with Advanced Augmentation & Fine-Tuning

### Changes
- Added advanced augmentations:
  - RandomResizedCrop
  - Horizontal Flip
  - Rotation
  - ColorJitter
- Increased epochs from 5 → 10
- Fine-tuned deeper layers (layer4)
- Used normalization

### Results

#### Validation Performance
- Best Validation Accuracy: 92.93%
- Epoch of Best Accuracy: 9

#### Test Performance
- Test Accuracy: 69.10%

#### Confusion Matrix
```
| Actual \ Predicted | glass | metal | paper | plastic |
|-------------------|-------|-------|-------|---------|
| glass             | 62    | 0     | 0     | 14      |
| metal             | 29    | 9     | 6     | 18      |
| paper             | 4     | 0     | 78    | 8       |
| plastic           | 9     | 0     | 5     | 59      |
```

#### Classification Report
```
Classification Report:
              precision    recall  f1-score   support

       glass       0.60      0.82      0.69        76
       metal       1.00      0.15      0.25        62
       paper       0.88      0.87      0.87        90
     plastic       0.60      0.81      0.69        73

    accuracy                           0.69       301
   macro avg       0.77      0.66      0.62       301
weighted avg       0.76      0.69      0.65       301
```

#### Classification Insights

- **Paper** remains the best-performing class (~87% F1-score)
- **Metal performance dropped significantly**:
  - Recall: 0.15 (very poor)
  - Model fails to correctly identify metal samples
- **Glass and plastic confusion persists**
- Model predicts "glass" too frequently (bias toward glass class)

---

### Observations

- Significant increase in validation accuracy (+15%) but decrease in test accuracy (-7.6%)
- Indicates **overfitting due to aggressive augmentations + deeper fine-tuning**
- Model memorizes training/validation patterns but fails to generalize to unseen data
- Class imbalance and feature similarity between classes (glass/plastic/metal) contribute to confusion

---

### Key Learning

- Higher validation accuracy does NOT guarantee better generalization
- Fine-tuning deeper layers without proper regularization can hurt performance
- Evaluation on test set is critical for true model assessment

---

### Next Steps

- Reduce augmentation intensity (especially RandomResizedCrop)
- Lower learning rate for fine-tuning
- Try partial freezing (freeze layer4 partially or completely)
- Add regularization (Dropout / Weight decay)
- Train for fewer epochs (early stopping)

## Experiment 3: Reduced Complexity (FC-only Training)

### Changes
- Removed deep fine-tuning (layer4 frozen)
- Trained only final classification layer
- Reduced learning rate to 0.0003
- Used moderate augmentations

### Results

#### Validation Performance
- Best Validation Accuracy: 77.10%
- Epoch of Best Accuracy: 9

#### Test Performance
- Test Accuracy: 67.44%

#### Confusion Matrix
```
| Actual \ Predicted | glass | metal | paper | plastic |
|-------------------|-------|-------|-------|---------|
| glass             | 56    | 6     | 6     | 8       |
| metal             | 18    | 28    | 4     | 12      |
| paper             | 5     | 1     | 82    | 2       |
| plastic           | 17    | 1     | 18    | 37      |
```

#### Classification Report
```
Classification Report:
              precision    recall  f1-score   support

       glass       0.58      0.74      0.65        76
       metal       0.78      0.45      0.57        62
       paper       0.75      0.91      0.82        90
     plastic       0.63      0.51      0.56        73

    accuracy                           0.67       301
   macro avg       0.68      0.65      0.65       301
weighted avg       0.68      0.67      0.66       301
```

#### Classification Insights

- **Paper remains strongest class (~0.82 F1-score)**
- **Metal recall dropped significantly (~0.45)**
- Model struggles with:
  - Plastic vs Glass
  - Plastic vs Paper
- Overall predictions are less confident compared to baseline

---

### Observations

- Removing fine-tuning reduced overfitting but led to **underfitting**
- Validation and test accuracy both decreased compared to baseline
- Model lacks capacity to learn deeper feature representations
- Performance is more stable but less accurate

---

### Key Learning

- Too much fine-tuning → overfitting  
- Too little fine-tuning → underfitting  
- Need a balance between feature learning and generalization  

---

### Conclusion

The FC-only approach reduces overfitting but significantly limits model performance. This confirms that some level of fine-tuning is necessary for optimal results.

---

### Next Steps

- Fine-tune only part of layer4 (partial unfreezing)
- Use smaller learning rate for deeper layers
- Introduce differential learning rates

## Experiment 4: Balanced Fine-Tuning (FC + Layer4 with Differential LR)

### Changes
- Fine-tuned both:
  - Final classification layer (fc)
  - Last convolutional block (layer4)
- Used differential learning rates:
  - FC layer: 0.0005
  - Layer4: 0.0001
- Used moderate augmentations and normalization
- Epochs: 10

### Results

#### Validation Performance
- Best Validation Accuracy: 93.27%
- Epoch of Best Accuracy: 8

#### Test Performance
- Test Accuracy: 71.76%

#### Confusion Matrix
```
| Actual \ Predicted | glass | metal | paper | plastic |
|-------------------|-------|-------|-------|---------|
| glass             | 64    | 0     | 1     | 11      |
| metal             | 18    | 8     | 3     | 33      |
| paper             | 0     | 0     | 84    | 6       |
| plastic           | 6     | 0     | 7     | 60      |
```
#### Classification Report
```
Classification Report:
              precision    recall  f1-score   support

       glass       0.73      0.84      0.78        76
       metal       1.00      0.13      0.23        62
       paper       0.88      0.93      0.91        90
     plastic       0.55      0.82      0.66        73

    accuracy                           0.72       301
   macro avg       0.79      0.68      0.64       301
weighted avg       0.79      0.72      0.67       301
```

#### Classification Insights

- **Paper performs best (~0.91 F1-score)**
- **Glass improved compared to previous experiments**
- **Plastic recall improved (~0.82)**
- **Metal remains the weakest class**:
  - Recall: 0.13 (very poor)
  - Model struggles to identify metal correctly
- Strong confusion between:
  - Metal ↔ Plastic
  - Metal ↔ Glass

---

### Observations

- Balanced fine-tuning improved test accuracy compared to:
  - Experiment 2 (+2.6%)
  - Experiment 3 (+4.3%)
- However, still worse than baseline (-5%)
- Validation accuracy remains high → slight overfitting still present
- Model is biased toward non-metal classes

---

### Key Learning

- Differential learning rates help stabilize fine-tuning
- Partial fine-tuning is better than:
  - Full fine-tuning (overfitting)
  - No fine-tuning (underfitting)
- However, class-specific issues (metal class) still limit performance

---

### Conclusion

Balanced fine-tuning improves performance over naive approaches but does not outperform the baseline model. Further improvements should focus on class imbalance handling and class-specific feature learning.

---

### Next Steps

- Improve metal class performance:
  - Use class weights in loss function
  - Try oversampling
- Try early stopping (best epoch ≈ 8)
- Experiment with learning rate scheduling
- Explore deeper architectures (ResNet34 / EfficientNet)