<h1 align="center">Final Result</h1>

## Classification Report Summary for MNIST

1. **For lenet-5 model**
    * **Accuracy:** 0.8805
    * **Macro Average:** Precision: 0.8790, Recall: 0.8782, F1-Score: 0.8782
    * **Weighted Average:** Precision: 0.8800, Recall: 0.8805, F1-Score: 0.8799

2. **For alexnet model**
    * **Accuracy:** 0.9797
    * **Macro Average:** Precision: 0.9798, Recall: 0.9797, F1-Score: 0.9796
    * **Weighted Average:** Precision: 0.9800, Recall: 0.9797, F1-Score: 0.9797

3. **For googlenet model**
    * **Accuracy:** 0.9949
    * **Macro Average:** Precision: 0.9949, Recall: 0.9949, F1-Score: 0.9949
    * **Weighted Average:** Precision: 0.9949, Recall: 0.9949, F1-Score: 0.9949

4. **For resnet18 model**
    * **Accuracy:** 0.6697
    * **Macro Average:** Precision: 0.7217, Recall: 0.6659, F1-Score: 0.6478
    * **Weighted Average:** Precision: 0.7257, Recall: 0.6697, F1-Score: 0.6528

### Conclusion

Based on the analysis of this second set of results Googlenet again achieves the highest performance with exceptional accuracy, confirming its strength on this dataset. Alexnet shows significantly improved and very strong performance, proving to be a highly capable architecture for this task, close behind Googlenet. Lenet-5's performance also improved but remains significantly lower than Alexnet and Googlenet, indicating its simpler structure is less effective for achieving top results. Resnet18 consistently exhibits the lowest performance across both tests, suggesting it is the least suitable or optimally configured architecture among the four for this specific task.

---
## Classification Report Summary for FMNIST

1. **For lenet-5 model**
    * **Accuracy:** 0.7798
    * **Macro Average:** Precision: 0.7748, Recall: 0.7798, F1-Score: 0.7748
    * **Weighted Average:** Precision: 0.7748, Recall: 0.7798, F1-Score: 0.7748

2. **For alexnet model**
    * **Accuracy:** 0.8514
    * **Macro Average:** Precision: 0.8522, Recall: 0.8514, F1-Score: 0.8493
    * **Weighted Average:** Precision: 0.8522, Recall: 0.8514, F1-Score: 0.8493

3. **For googlenet model**
    * **Accuracy:** 0.9303
    * **Macro Average:** Precision: 0.9306, Recall: 0.9303, F1-Score: 0.9300
    * **Weighted Average:** Precision: 0.9306, Recall: 0.9303, F1-Score: 0.9300

4. **For resnet18 model**
    * **Accuracy:** 0.6707
    * **Macro Average:** Precision: 0.6716, Recall: 0.6707, F1-Score: 0.6622
    * **Weighted Average:** Precision: 0.6716, Recall: 0.6707, F1-Score: 0.6622

### Conclusion

Based on the analysis of the results Googlenet demonstrated superior performance across all metrics, indicating its complex architecture with inception modules is highly effective for this task. Alexnet showed moderate performance, better than simpler models but not reaching Googlenet's level. Lenet-5, a foundational architecture, performed the lowest among the older models, suggesting its simplicity is a limitation here. Surprisingly, Resnet18 underperformed significantly, highlighting that performance can vary depending on the specific architecture variant and dataset characteristics.

---
## Classification Report Summary for CIFAR-10

1. **For lenet-5 model**
    * **Accuracy:** 0.4032
    * **Macro Average:** Precision: 0.3944, Recall: 0.4032, F1-Score: 0.3934
    * **Weighted Average:** Precision: 0.3944, Recall: 0.4032, F1-Score: 0.3934
2. **For alexnet model**
    * **Accuracy:** 0.4647
    * **Macro Average:** Precision: 0.4649, Recall: 0.4647, F1-Score: 0.4447
    * **Weighted Average:** Precision: 0.4649, Recall: 0.4647, F1-Score: 0.4447
3. **For googlenet model**
    * **Accuracy:** 0.7846
    * **Macro Average:** Precision: 0.7910, Recall: 0.7846, F1-Score: 0.7835
    * **Weighted Average:** Precision: 0.7910, Recall: 0.7846, F1-Score: 0.7835
4. **For resnet18 model**
    * **Accuracy:** 0.4288
    * **Macro Average:** Precision: 0.4299, Recall: 0.4288, F1-Score: 0.4213
    * **Weighted Average:** Precision: 0.4299, Recall: 0.4288, F1-Score: 0.4213

### Conclusion

Based on the performance results for CIFAR-10 Googlenet demonstrates superior performance, indicating its advanced architecture is highly effective for this more complex image dataset. Alexnet performs moderately well but is significantly outperformed by Googlenet. Resnet18 shows better capability than Lenet-5 on CIFAR-10, suggesting some advantage from its structure, although it lags behind Alexnet and Googlenet. Lenet-5 proves least effective on CIFAR-10, highlighting its limitations on datasets more complex than grayscale digits.
