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