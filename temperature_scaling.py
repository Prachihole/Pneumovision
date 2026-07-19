"""
Temperature Scaling wrapper.
Learns a single scalar T that smooths overconfident softmax outputs:
    calibrated_probs = softmax(logits / T)
"""

import torch
import torch.nn as nn
import torch.optim as optim


class ModelWithTemperature(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, x):
        logits = self.model(x)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    def set_temperature(self, valid_loader, device="cpu"):
        self.to(device)
        nll_criterion = nn.CrossEntropyLoss().to(device)

        logits_list = []
        labels_list = []
        self.model.eval()
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs = inputs.to(device)
                logits = self.model(inputs)
                logits_list.append(logits.cpu())
                labels_list.append(labels)

        logits = torch.cat(logits_list).to(device)
        labels = torch.cat(labels_list).to(device)

        before_nll = nll_criterion(logits, labels).item()
        print(f"Before temperature scaling -> NLL: {before_nll:.4f}")

        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval_step():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss

        optimizer.step(eval_step)

        after_nll = nll_criterion(self.temperature_scale(logits), labels).item()
        print(f"After temperature scaling  -> NLL: {after_nll:.4f}")
        print(f"Learned temperature: {self.temperature.item():.4f}")

        return self