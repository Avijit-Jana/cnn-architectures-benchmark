<h1 align="center">Final Result</h1>

## Classification Report Summary for MNIST

1. **For lenet-5 model**
    * **Accuracy:** 0.8453
    * **Macro Average:** Precision: 0.8434, Recall: 0.8419, F1-Score: 0.8414
    * **Weighted Average:** Precision: 0.8449, Recall: 0.8453, F1-Score: 0.8439

2. **For alexnet model**
    * **Accuracy:** 0.9597
    * **Macro Average:** Precision: 0.9602, Recall: 0.9591, F1-Score: 0.9593
    * **Weighted Average:** Precision: 0.9605, Recall: 0.9597, F1-Score: 0.9597

3. **For googlenet model**
    * **Accuracy:** 0.9910
    * **Macro Average:**  Precision: 0.9911, Recall: 0.9910, F1-Score: 0.9910
    * **Weighted Average:** Precision: 0.9911, Recall: 0.9910, F1-Score: 0.9910

4. **For resnet18 model**
    * **Accuracy:** 0.5182
    * **Macro Average:** Precision: 0.5229, Recall: 0.5091, F1-Score: 0.4899
    * **Weighted Average:** Precision: 0.5262, Recall: 0.5182, F1-Score: 0.4963

5. **For xception model**
    * **Accuracy:** 0.9807
    * **Macro Average:** Precision: 0.9806, Recall: 0.9806, F1-Score: 0.9806
    * **Weighted Average:** Precision: 0.9807, Recall: 0.9807, F1-Score: 0.9807

<table>
    <tr>
    <td><img src="assets\images\MNIST_accuracy.png" alt="accuracy" width="300"></td>
    <td><img src="assets\images\MNIST_macro_average.png" alt="macro_average" width="300"></td>
    <td><img src="assets\images\MNIST_weighted_average.png" alt="weighted_average" width="300"></td>
    </tr>
</table>

### Conclusion

Among the five architectures trained from scratch on MNIST, GoogLeNet achieved the highest accuracy at 99.10%, followed by Xception at 98.07% and AlexNet at 95.97%. LeNet-5—a relatively shallow network—reached 84.53%, while ResNet-18, despite its success on larger datasets, underperformed here with only 51.82%, indicating a likely training misconfiguration. Both GoogLeNet and Xception demonstrated very balanced performance across all digit classes (macro and weighted averages nearly identical), whereas LeNet-5 and AlexNet showed modest but consistent gains over simpler designs. In short, efficient module design (GoogLeNet’s inception blocks) and depthwise separable convolutions (Xception) proved most effective for MNIST, while ResNet-18’s atypically low result flags a need to revisit its training setup.

---

## Classification Report Summary for FMNIST

1. **For lenet-5 model**
    * **Accuracy:** 0.7516
    * **Macro Average:** Precision: 0.7436, Recall: 0.7516, F1-Score: 0.7458
    * **Weighted Average:** Precision: 0.7436, Recall: 0.7516, F1-Score: 0.7458

2. **For alexnet model**
    * **Accuracy:** 0.7968
    * **Macro Average:** Precision: 0.7906, Recall: 0.7968, F1-Score: 0.7876
    * **Weighted Average:** Precision: 0.7906, Recall: 0.7968, F1-Score: 0.7876

3. **For googlenet model**
    * **Accuracy:** 0.9117
    * **Macro Average:** Precision: 0.9131, Recall: 0.9117, F1-Score: 0.9121
    * **Weighted Average:** Precision: 0.9131, Recall: 0.9117, F1-Score: 0.9121

4. **For resnet18 model**
    * **Accuracy:** 0.6653
    * **Macro Average:** Precision: 0.6805, Recall: 0.6653, F1-Score: 0.6579
    * **Weighted Average:** Precision: 0.6805, Recall: 0.6653, F1-Score: 0.6579

5. **For xception model**
    * **Accuracy:** 0.8630
    * **Macro Average:** Precision: 0.8616, Recall: 0.8630, F1-Score: 0.8598
    * **Weighted Average:** Precision: 0.8616, Recall: 0.8630, F1-Score: 0.8598

<table>
    <tr>
    <td><img src="assets\images\FMNIST_accuracy.png" alt="accuracy" width="300"></td>
    <td><img src="assets\images\FMNIST_macro_average.png" alt="macro_average" width="300"></td>
    <td><img src="assets\images\FMNIST_weighted_average.png" alt="weighted_average" width="300"></td>
    </tr>
</table>    

### Conclusion

Based on the analysis of the results Googlenet demonstrated superior performance across all metrics, indicating its complex architecture with inception modules is highly effective for this task. Alexnet showed moderate performance, better than simpler models but not reaching Googlenet's level. Lenet-5, a foundational architecture, performed the lowest among the older models, suggesting its simplicity is a limitation here. Surprisingly, Resnet18 underperformed significantly, highlighting that performance can vary depending on the specific architecture variant and dataset characteristics.

---

## Classification Report Summary for CIFAR-10

1. **For lenet-5 model**
    * **Accuracy:** 0.3669
    * **Macro Average:** Precision: 0.3555, Recall: 0.3669, F1-Score: 0.3440
    * **Weighted Average:** Precision: 0.3555, Recall: 0.3669, F1-Score: 0.3440
2. **For alexnet model**
    * **Accuracy:** 0.2727
    * **Macro Average:** Precision: 0.2392, Recall: 0.2727, F1-Score: 0.2261
    * **Weighted Average:** Precision: 0.2392, Recall: 0.2727, F1-Score: 0.2261
3. **For googlenet model**
    * **Accuracy:** 0.7461
    * **Macro Average:** Precision: 0.7593, Recall: 0.7461, F1-Score: 0.7468
    * **Weighted Average:** Precision: 0.7593, Recall: 0.7461, F1-Score: 0.7468
4. **For resnet18 model**
    * **Accuracy:** 0.3757
    * **Macro Average:** Precision: 0.3813, Recall: 0.3757, F1-Score: 0.3592
    * **Weighted Average:** Precision: 0.3813, Recall: 0.3757, F1-Score: 0.3592
5. **For xception model**
    * **Accuracy:** 0.5111
    * **Macro Average:** Precision: 0.5224, Recall: 0.5111, F1-Score: 0.5019
    * **Weighted Average:** Precision: 0.5224, Recall: 0.5111, F1-Score: 0.5019

<table>
    <tr>
    <td><img src="assets\images\CIFAR10_accuracy.png" alt="accuracy" width="300"></td>
    <td><img src="assets\images\CIFAR10_macro_average.png" alt="macro_average" width="300"></td>
    <td><img src="assets\images\CIFAR10_weighted_average.png" alt="weighted_average" width="300"></td>
    </tr>
</table>    

### Conclusion

GoogLeNet outperforms all other models on CIFAR-10 with about 74.6% accuracy and strong F1 scores, followed by Xception at roughly 51.1%, showing that advanced modules like Inception blocks and depthwise-separable convolutions with residuals make a big difference; in contrast, ResNet-18 (≈37.6%) and LeNet-5 (≈36.7%) perform similarly and lag behind, while AlexNet (≈27.3%) fares worst. This suggests that deeper, specialized architectures excel on CIFAR-10, and that adding stronger augmentations, better LR schedules, or pretraining could further boost ResNet-18 or Xception, though GoogLeNet remains the immediate top choice.
