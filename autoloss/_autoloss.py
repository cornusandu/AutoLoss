import torch
import torch.nn as nn
import torch.optim as optim
import math
from typing import Optional

# Define the loss predictor as a simple feedforward network.
class LossPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super(LossPredictor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)


# Define the model "Bob" (a simple neural network).
class Bob(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Bob, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


# Define the AutoLoss optimizer that leverages the loss predictor.
class AutoLoss:
    def __init__(self, bob, loss_fn, candidate_noise=0.01,
                 patience=20, lr=0.001, hidden_dim = 32):

        flat_param_dim = sum(p.numel() for p in bob.parameters())

        loss_predictor = LossPredictor(input_dim=flat_param_dim, hidden_dim=hidden_dim)
        sgd_optimizer = optim.SGD(bob.parameters(), lr=lr)

        self.bob = bob
        self.loss_predictor = loss_predictor
        self.predictor_optimizer = optim.Adam(loss_predictor.parameters(), lr=0.001 if patience <= 100 else 0.0001)
        self.sgd_optimizer = sgd_optimizer
        self.loss_fn = loss_fn
        self.candidate_noise = candidate_noise
        self.patience = patience
        self.patience_percent = 70
        self.no_improve_count = 0
        self.use_sgd = False

    def get_flat_params(self):
        # Flatten all parameters of Bob into a single vector.
        return torch.cat([p.data.view(-1) for p in self.bob.parameters()])

    def set_flat_params(self, flat_params):
        # Update Bob's parameters using a flat vector.
        current_idx = 0
        for p in self.bob.parameters():
            numel = p.numel()
            p.data.copy_(flat_params[current_idx:current_idx + numel].view_as(p))
            current_idx += numel

    def step(self, x: Optional[torch.Tensor], target):
        # If the predictor has not found improvements for several iterations,
        # switch to standard SGD.
        if self.use_sgd:
            self.sgd_optimizer.zero_grad()
            if x is not None:
                output = self.bob(x)
            else:
                output = self.bob()
            loss = self.loss_fn(output, target)
            loss.backward()
            self.sgd_optimizer.step()
            return loss.item()

        # Generate candidate parameters by adding a small Gaussian noise.
        current_params = self.get_flat_params()
        candidate_params = current_params + torch.randn_like(current_params) * self.candidate_noise

        # Use the loss predictor to estimate the candidate's loss.
        self.loss_predictor.eval()
        with torch.no_grad():
            predicted_loss = self.loss_predictor(candidate_params.unsqueeze(0)).item()

        # Compute the actual loss using the current parameters.
        self.bob.eval()
        with torch.no_grad():
            if x is not None:
                current_output = self.bob(x)
            else:
                current_output = self.bob()
            current_loss = self.loss_fn(current_output, target).item()

        # Accept candidate parameters if the predicted loss is lower.
        if predicted_loss < current_loss:
            self.set_flat_params(candidate_params)
            improvement = current_loss - predicted_loss
        else:
            improvement = 0

        # Train the loss predictor using the current parameters and actual loss.
        self.loss_predictor.train()
        self.predictor_optimizer.zero_grad()
        flat_params = self.get_flat_params().unsqueeze(0)
        predicted = self.loss_predictor(flat_params)
        predictor_loss = (predicted - torch.tensor([[current_loss]], dtype=predicted.dtype)) ** 2
        predictor_loss.mean().backward()
        self.predictor_optimizer.step()

        # Update the count of iterations with no improvement.
        if improvement <= 1e-6:
            self.no_improve_count += 1
        else:
            self.no_improve_count = 0

        # If no improvement has been observed for a number of iterations, switch to SGD.
        if self.no_improve_count >= self.patience:
            # print("[Debug] Starting to use SGD!")
            self.use_sgd = True

        # Perform a training step for Bob using the current parameters.
        self.bob.train()
        self.sgd_optimizer.zero_grad()
        output = self.bob(x)
        loss = self.loss_fn(output, target)
        loss.backward()
        self.sgd_optimizer.step()

        return loss.item()


# Example usage:
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    EPOCHS = 1000
    SGD_PER = ( 40 ) * 100 / EPOCHS

    # Create dummy input and target data.
    x = torch.randn(10, 5)  # 10 samples with 5 features each
    target = torch.randn(10, 1)

    # Initialize Bob.
    bob = Bob(input_dim=5, hidden_dim=64, output_dim=1)
    # Determine the dimensionality of the flattened parameter vector.

    # Define a loss function.
    loss_fn = nn.MSELoss()

    # Initialize AutoLoss with a candidate noise level and a patience threshold.
    auto_loss = AutoLoss(bob, loss_fn, candidate_noise=0.01,
                         patience=math.floor(SGD_PER/100*EPOCHS))

    losses = []

    epoch_amount_switch_to_SGD = None

    for epoch in range(EPOCHS):
        loss_value = auto_loss.step(x, target)
        losses.append(loss_value)
        print(f"Epoch {epoch}: Loss = {loss_value:.4f}")

        # Setează epoca de switch la SGD o singură dată.
        if auto_loss.use_sgd and epoch_amount_switch_to_SGD is None:
            # epoch + 1 dacă vrei să marchezi prima epocă *complet* cu SGD
            epoch_amount_switch_to_SGD = epoch

    epochs = list(range(EPOCHS))

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, label='Loss', marker='o', markersize=2, linewidth=1)

    # Draw a vertical blue dashed line at epoch 35 to indicate the start of SGD usage
    plt.axvline(x=epoch_amount_switch_to_SGD or 0, color='blue', linestyle='--', label='Switch to SGD')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss vs. Epoch with SGD Transition')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
