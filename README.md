# AutoLoss
AutoLoss is an optimizer that uses Artificial Intelligence to optimize your neural networks as well as possible.
## Usage
First, import the package and other dependencies:
```python
from AutoLoss import *
import torch
import torch.nn as nn
```
Then, create your AI Model. For the sake of simplicity, let's just name it `model` for this example.

Next, initialize a loss function. For this example, we'll use CrossEntropyLoss:
```python
loss_fn = nn.CrossEntropyLoss()
```

Afterwards, create an instance of the AutoLoss optimizer

```python
optimizer = AutoLoss(model, loss_fn, patience=25)
```
Note: For more advanced details on the `patience` argument, check **Internal Functioning**.

Now, to train the model, just call `optimizer.step(x, target)`, where `x` is the input data and `target` is the expected output data.

## Internal Functioning
- AutoLoss uses its own AI model to predict what parameters would work best for your neural network, therefore constantly getting better at predicting better parameters and training your AI better.
- If AutoLoss's AI can not find any better parameters, it switches to SGD (Stochastic Gradient Descent).
- The amount of tries AutoLoss's AI has at finding better parameters until it switches to SGD is equal to the `patience` parameter from earlier.