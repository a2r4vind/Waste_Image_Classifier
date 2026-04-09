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
- Can also use weight decay

## Experiment 4: ResNet18 + Fine-Tuning + Differential Learning Rates + Weight Decay

### Objective
Improve model generalization and stability by:
- Applying differential learning rates
- Adding weight decay regularization

---

### Changes from Experiment 2
- Used differential learning rates:
  - FC layer → lr = 0.0005
  - Layer4 → lr = 0.0001
- Added weight decay = 1e-4
- Same augmentation as Experiment 2

---

### Model Configuration
- Architecture: ResNet18 (pretrained on ImageNet)
- Trainable layers:
  - Fully Connected (FC)
  - Layer4
- Frozen layers:
  - All earlier layers

---

### Training Results
- Best Validation Accuracy: 91.25%
- Best Epoch: ~7–9
- Final Training Accuracy: ~98%

---

### Test Results
- Test Accuracy: 90.37%

---

#### Confusion Matrix
```
| Actual \ Predicted | glass | metal | paper | plastic |
|--------------------|-------|-------|-------|---------|
| metal              | 70    | 5     | 0     | 1       |
| glass              | 6     | 56    | 0     | 0       |
| paper              | 2     | 1     | 87    | 0       |
| plastic            | 6     | 5     | 3     | 59      |
```

---

#### Classification Report
```
Classification Report:
              precision    recall  f1-score   support

       glass       0.83      0.92      0.88        76
       metal       0.84      0.90      0.87        62
       paper       0.97      0.97      0.97        90
     plastic       0.98      0.81      0.89        73

    accuracy                           0.90       301
   macro avg       0.90      0.90      0.90       301
weighted avg       0.91      0.90      0.90       301
```

---
## Error Analysis 

### Total Misclassified Samples
- **29 / 301**

### Error Distribution (True → Predicted : Sample Count)
- metal   → glass   : 6
- plastic → glass   : 6
- glass   → metal   : 5
- plastic → metal   : 5
- plastic → paper   : 3
- paper   → glass   : 2
- glass   → plastic : 1
- paper   → metal   : 1

---

### Observations in error analysis

1. **Glass vs Metal Confusion**
   - Frequent bidirectional errors (glass ↔ metal)
   - Likely due to:
     - Reflective surfaces
     - Similar textures (shiny/transparent materials)

2. **Plastic misclassified as Glass/Metal**
   - Plastic bottles with transparency resemble glass
   - Rigid plastic objects resemble metal containers

3. **Paper performs best**
   - Highest precision and recall (0.97)
   - Likely due to distinct texture and appearance

4. **Plastic Recall is lower (0.81)**
   - Indicates model struggles to consistently identify plastic

---

### Visual Error Inspection

- Misclassified samples saved at:

results/exp4_resnet18_balanced_finetune_differential_lrs_weight_decay_misclassified.png


- Key patterns observed:
  - Transparent objects → confusion (glass vs plastic)
  - Shiny/reflective surfaces → confusion (glass vs metal)
  - Background clutter impacts classification

---

### Key Observations
- Differential learning rates improved stability of fine-tuning
- Weight decay reduced overfitting
- Metal class performance improved significantly
- Plastic still has minor confusion with other classes

---

### Comparison with Previous Experiments
```
| Experiment |           Strategy              | Test Accuracy |
|------------|---------------------------------|---------------|
| Exp1       | Baseline (FC only)              | 79.07%        |
| Exp2       | Augmentation + Layer4 fine-tune | 90.03%        |
| Exp3       | Reduced LR (FC only)            | 75.75%        |
| Exp4       | Differential LR + Weight Decay  | **90.37%**    |
```
---

### Conclusion
- Best performing model so far
- Balanced performance across all classes
- Selected as current **champion model**

## Champion Model

**Experiment:** Exp4 (ResNet18 + Differential LR + Weight Decay)  
**Test Accuracy:** 90.37%  

This model provides the best balance between performance and generalization.

### Next Steps

- Explore deeper architectures (ResNet34 / EfficientNet)

## Experiment 5: ResNet34 + Fine-Tuning + Differential Learning Rates + Weight Decay

### Objective
To evaluate whether a deeper architecture (ResNet34) improves performance compared to ResNet18, while using:
- Data augmentation
- Fine-tuning (Layer4 + FC)
- Differential learning rates
- Weight decay for regularization

---

### Model Configuration
- Model: ResNet34 (pretrained on ImageNet)
- Trainable Layers:
  - Fully Connected (FC)
  - Layer4 (fine-tuning)
- Optimizer: Adam
  - FC layer → LR = 0.0005
  - Layer4 → LR = 0.0001
  - Weight Decay = 1e-4
- Loss Function: CrossEntropyLoss
- Batch Size: 32
- Epochs: 10

---

### Data Augmentation
- RandomResizedCrop (224)
- Horizontal Flip (p=0.5)
- Rotation (20°)
- ColorJitter (brightness, contrast, saturation = 0.3)
- Normalization (ImageNet stats)

---

### Training Results
- Best Validation Accuracy: 91.25% (Epoch 4)
- Training Accuracy reached ~98% → slight overfitting observed after epoch 4

---

### Test Results
- Test Accuracy: 90.03%

#### Confusion Matrix
```
| Actual \ Predicted | glass | metal | paper | plastic |
|--------------------|-------|-------|-------|---------|
| metal              | 71    | 4     | 0     | 1       |
| glass              | 6     | 54    | 1     | 1       |
| paper              | 2     | 0     | 86    | 2       |
| plastic            | 8     | 2     | 3     | 60      |
```

---

#### Classification Report
```
Classification Report:
              precision    recall  f1-score   support

       glass       0.82      0.93      0.87        76
       metal       0.90      0.87      0.89        62
       paper       0.96      0.96      0.96        90
     plastic       0.94      0.82      0.88        73

    accuracy                           0.90       301
   macro avg       0.90      0.90      0.90       301
weighted avg       0.90      0.90      0.90       301
```

---
### Observations
- Significant improvement over baseline (EXP1 ~79% → EXP5 ~90%)
- Performance comparable to EXP4 (ResNet18 tuned model)
- Paper class remains easiest (highest precision & recall)
- Plastic recall improved slightly, but still lower than other classes
- Model shows mild overfitting after epoch 4

---

### Comparison with Previous Experiments
```
| Experiment |   Model  |           Strategy                   | Test Accuracy |
|------------|----------|--------------------------------------|---------------|
| Exp1       | ResNet18 |Baseline (FC only)                    | 79.07%        |
| Exp2       | ResNet18 |Augmentation + Layer4 fine-tune       | 90.03%        |
| Exp3       | ResNet18 |Reduced LR (FC only)                  | 75.75%        |
| Exp4       | ResNet18 |Differential LR + Weight Decay        | **90.37%**    |
| Exp5       | ResNet34 |Deeper model + same strategy as EXP4  | 90.03%        |
```

---

### Classification Insights
- Increasing model depth (ResNet34) did NOT significantly outperform ResNet18
- Proper fine-tuning + optimization strategy matters more than model size
- EXP4 remains the best-performing experiment

---

## Final Conclusion

- Baseline model (EXP1) provided a strong starting point (~79%)
- Data augmentation + fine-tuning (EXP2) gave the biggest performance boost
- Reducing learning rate without fine-tuning (EXP3) hurt performance
- Differential learning rates + weight decay (EXP4) achieved the best results (~90.37%)
- Increasing model depth (EXP5) did not significantly improve performance

### Best Model:
**EXP4 - ResNet18 with differential learning rates and weight decay**

### Key Takeaways:
- Fine-tuning deeper layers is critical for performance
- Data augmentation significantly improves generalization
- Differential learning rates stabilize training
- Bigger models are not always better