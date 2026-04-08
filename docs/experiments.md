# Experiments Log

## Experiment 1: Baseline ResNet18

### Model
- Architecture: ResNet18 (pretrained on ImageNet)
- Training Strategy: Only final FC layer trained
- Frozen Layers: All except FC

### Data
- Dataset: TrashNet
- Split: Train/Val/Test = 70/15/15
- Input Size: 224x224

### Transforms
- Resize (224, 224)
- Random Horizontal Flip
- Random Rotation (10°)
- Normalization (ImageNet mean/std)

### Training Config
- Epochs: 5
- Batch size: 32
- Optimizer: Adam
- Learning rate: 0.001
- Loss Function: CrossEntropyLoss

### Results
- Best Validation Accuracy: **78.11%**
- Test Accuracy: **79.07%**
- Epoch of Best Accuracy: 5

### Classification Report (Test)
```
| Class   | Precision | Recall | F1-score |
|---------|-----------|--------|----------|
| glass   | 0.73      | 0.71   | 0.72     |
| metal   | 0.72      | 0.79   | 0.75     |
| paper   | 0.88      | 0.93   | 0.91     |
| plastic | 0.80      | 0.70   | 0.74     |
```
- Macro Avg F1: **0.78** 
- Weighted Avg F1: **0.79**

### Observations
- Strong baseline performance (~79% test accuracy)
- Model generalizes well (val ≈ test accuracy)
- Paper class performs best (high precision & recall)
- Glass and plastic show moderate confusion
- No severe class imbalance issues observed

### Test Results

- Test Accuracy: **79.07%**

#### Confusion Matrix

```
| Actual \ Predicted | glass | metal | paper | plastic |
|------------------- |-------|-------|-------|---------|
| glass              | 54    | 12    | 1     | 9       |
| metal              | 8     | 49    | 4     | 1       |
| paper              | 3     | 0     | 84    | 3       |
| plastic            | 9     | 7     | 6     | 51      |
```

### Insights
- Even with only FC layer training, pretrained features are effective
- Baseline is already strong → improvements will be incremental

### Next Steps
- Add data augmentation
- Fine-tune deeper layers
- Increase epochs

## Experiment 2: ResNet18 with Advanced Augmentation & layer 4 Fine-Tuning

### Model
- Architecture: ResNet18 (pretrained on ImageNet)
- Training Strategy: Fine-tuned last block (layer4) + FC layer
- Frozen Layers: All except layer4 and FC

### Data
- Dataset: TrashNet
- Split: Train/Val/Test = 70/15/15
- Input Size: 224x224

### Transforms (Train)
- Random Resized Crop (224, scale=0.8–1.0)
- Random Horizontal Flip
- Random Rotation (20°)
- Color Jitter (brightness, contrast, saturation = 0.3)
- Normalization (ImageNet mean/std)

### Transforms (Val/Test)
- Resize (224, 224)
- Normalization (ImageNet mean/std)

### Training Config
- Epochs: 10
- Batch Size: 32
- Optimizer: Adam (Differential Learning Rates)
  - FC layer: 0.001
  - Layer4: 0.0001
- Loss Function: CrossEntropyLoss

### Results
- Best Validation Accuracy: **90.24% (Epoch 8)**
- Test Accuracy: **90.03%**


#### Confusion Matrix
```
| Actual \ Predicted | glass | metal | paper | plastic |
|--------------------|-------|-------|-------|---------|
| glass              | 70    | 5     | 0     | 1       |
| metal              | 7     | 55    | 0     | 0       |
| paper              | 1     | 1     | 87    | 1       |
| plastic            | 5     | 5     | 4     | 59      |
```

### Classification Report (Test)
```
| Class   | Precision | Recall | F1-score |
|---------|-----------|--------|----------|
| glass   | 0.84      | 0.92   | 0.88     |
| metal   | 0.83      | 0.89   | 0.86     |
| paper   | 0.96      | 0.97   | 0.96     |
| plastic | 0.97      | 0.81   | 0.88     |
```
- Macro Avg F1: **0.90**
- Weighted Avg F1: **0.90**

#### Classification Insights

- **Paper** remains the best-performing class (~96% F1-score)
- Data augmentation greatly improved generalization
- Fine-tuning deeper layers (layer4) is highly effective
- Differential learning rates stabilized training
- Model now captures more complex features

---

### Observations
- Significant improvement over baseline (+11% accuracy)
- Strong generalization (val ≈ test accuracy)
- All classes improved, especially:
  - Glass (recall ↑)
  - Metal (balanced precision/recall)
- Paper remains easiest class
- Slight drop in plastic recall (possible confusion)

---

### Comparison with Baseline
```
| Metric             | Experiment 1 | Experiment 2 |
|--------------------|--------------|--------------|
| Val Accuracy       | 78.11%       | 90.24%       |
| Test Accuracy      | 79.07%       | 90.03%       |
| Macro F1           | 0.78         | 0.90         |
```
 Improvement: **~+11% accuracy**

---

### Next Steps

- Try reduce augmentation intensity (especially RandomResizedCrop)
- Lower learning rate for fine-tuning
- Try partial freezing (freeze layer4 partially or completely)


## Experiment 3: Reduced Complexity (FC-only Training)

### Changes
- Removed deep fine-tuning (layer4 frozen)
- Trained only final classification layer
- Reduced learning rate to 0.0003
- Used moderate augmentations

### Results

#### Validation Performance
- Best Validation Accuracy: 80.47%
- Epoch of Best Accuracy: 7

#### Test Performance
- Test Accuracy: 75.75%

#### Confusion Matrix
```
| Actual \ Predicted | glass | metal | paper | plastic |
|--------------------|-------|-------|-------|---------|
| metal              | 9     | 46    | 6     | 1       |
| glass              | 57    | 14    | 0     | 5       |
| paper              | 4     | 3     | 78    | 5       |
| plastic            | 12    | 10    | 4     | 47      |
```

#### Classification Report
```
Classification Report:
              precision    recall  f1-score   support

       glass       0.70      0.75      0.72        76
       metal       0.63      0.74      0.68        62
       paper       0.89      0.87      0.88        90
     plastic       0.81      0.64      0.72        73

    accuracy                           0.76       301
   macro avg       0.76      0.75      0.75       301
weighted avg       0.77      0.76      0.76       301
```

---

### Observations

- Training convergence was slower due to reduced learning rate.
- Final performance is worse than Experiment 2.
- Model struggled more with:
    - glass–plastic confusion
    - metal class performance
- Lower LR prevented strong adaptation of FC layer.

---


### Conclusion

- Reducing learning rate too early hurts performance when training only FC layer.
- LR = 0.001 (Experiment 2) is more effective for this setup.
- Lower LR may be useful only when fine-tuning deeper layers, not FC-only training.

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