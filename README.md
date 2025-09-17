# CNN-MNIST (Params: <25K)

This project explores training Convolutional Neural Networks (CNNs) on the MNIST dataset, focusing on achieving high accuracy (>95%) with a minimal number of parameters. The experiments are organized into four main notebook iterations, each exploring different model architectures, hyperparameters, and training strategies. Each notebook may contain multiple sub-iterations or trials, with results and observations documented for each.

## Iteration Summaries

### Iteration 1: Baseline Model
- **Model Architecture:**
  - 4 convolutional layers: [Conv2d(1,32,3), Conv2d(32,64,3), Conv2d(64,128,3), Conv2d(128,256,3)]
  - 2 fully connected layers: [Linear(4096,50), Linear(50,10)]
  - **Total Parameters:** ~593,200
- **Training Details:**
  - Optimizer: SGD, lr=10.01, momentum=0.9
  - Batch size: 512
  - Data augmentation: CenterCrop, Resize, RandomRotation, Normalization
- **Results:**
  - Training and test accuracy did not improve (accuracy ~9.9%).
  - Model is over-parameterized for the task and did not converge.
- **Sub-Iterations/Trials:**
  - Batch size 256: Accuracy ~9.9%
  - Batch size 1024: Accuracy ~9.9%
  - Other minor changes: No significant improvement

### Iteration 2: Parameter Tuning
- **Model Architecture:**
  - Same as Iteration 1
- **Training Details:**
  - Batch size increased to 1024
  - Optimizer: SGD, lr=1, momentum=0.9 (also tried lr=0.5)
- **Results:**
  - Model still failed to converge (accuracy ~9.9%).
  - Large model size and high learning rate likely caused instability.
- **Sub-Iterations/Trials:**
  - Learning rate 1.0: Accuracy ~9.9%
  - Learning rate 0.5: Accuracy ~9.9%
  - Other optimizer settings: No successful convergence

### Iteration 3: Reduced Model Size
- **Model Architecture:**
  - 3 convolutional layers: [Conv2d(1,16,3), Conv2d(16,32,3), Conv2d(32,64,3)]
  - 2 fully connected layers: [Linear(576,65), Linear(65,10)]
  - **Total Parameters:** ~61,461
- **Training Details:**
  - Optimizer: Adam, lr=0.01, weight_decay=1e-4 (also tried lr=0.001, weight_decay=1e-4 and lr=0.01, weight_decay=1e-3)
  - Batch size: 1024
- **Results:**
  - Model converged and achieved accuracy >95%.
  - Reducing model size and using Adam optimizer improved results.
- **Sub-Iterations/Trials:**
  - Adam, lr=0.01, weight_decay=1e-4: Accuracy >95%
  - Adam, lr=0.001, weight_decay=1e-4: Accuracy >95%
  - Adam, lr=0.01, weight_decay=1e-3: Accuracy >95%

### Iteration 4: Minimal Model (<25K Parameters)
- **Model Architecture:**
  - 2 convolutional layers: [Conv2d(1,16,3), Conv2d(16,32,3)]
  - 2 fully connected layers: [Linear(288,65), Linear(65,10)]
  - **Total Parameters:** ~24,245
- **Training Details:**
  - Optimizer: Adam, lr=0.01, weight_decay=1e-4
  - Batch size: 1024
- **Results:**
  - Model achieved >95% accuracy with fewer than 25,000 parameters.
  - Demonstrates that a compact CNN can perform well on MNIST with proper regularization and training.
- **Sub-Iterations/Trials:**
  - Final architecture and hyperparameters: Accuracy >95%

## Key Takeaways
- Overly large models can fail to converge on simple datasets like MNIST.
- Reducing model size and using Adam optimizer with appropriate regularization leads to better results.
- It is possible to achieve >95% accuracy on MNIST with fewer than 25,000 parameters.

## Notebooks
- `mnist-iteration1.ipynb`: Baseline large model, unsuccessful training, and several sub-iterations/trials.
- `mnist-iteration2.ipynb`: Parameter tuning, multiple sub-iterations, still unsuccessful.
- `mnist-iteration3.ipynb`: Reduced model, several sub-iterations, successful training.
- `mnist-iteration4.ipynb`: Minimal model, final sub-iterations, successful and efficient.

## Final Model Architecture (Iteration 4)

Below is a horizontal ASCII/text diagram of the final minimal CNN architecture used in Iteration 4:

```
Input (1x28x28) → Conv2d(1,16,3x3) → MaxPool2d(2x2) → Conv2d(16,32,3x3) → MaxPool2d(3x3) → Flatten (288) → Linear(288,65) → ReLU → Linear(65,10) → LogSoftmax (10 classes)
```

This compact architecture achieves high accuracy on MNIST with fewer than 25,000 parameters.