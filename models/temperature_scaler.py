# Adapted from https://github.com/gpleiss/temperature_scaling
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

sys.path.insert(0, '../optimizers')

from optimizers.ece_loss import ECELossBinary
from optimizers.focal_loss import FocalLossBinary

class TemperatureScaler(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the sigmoid or log sigmoid!
    """
    def __init__(self, model, device='cpu'):
        super(TemperatureScaler, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        self.device = device
        self.sigmoid = nn.Sigmoid()

    def forward(self, batch):
        logits = self.model(batch)

        return self.sigmoid(self.scale_temperature(logits))

    def scale_temperature(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))

        return logits / temperature

    # This function probably should live outside of this class, but whatever
    def set_temperature(self, val_loader, alpha=0.25, gamma=2):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize FL.
        val_loader (DataLoader): validation set loader
        """
        self.to(self.device)
        fl_criterion = FocalLossBinary(
            alpha=alpha,
            gamma=gamma,
            logits=True
        ).to(self.device)
        ece_criterion = ECELossBinary(logits=True).to(self.device)

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for batch, label in val_loader:
                batch = batch.to(self.device)
                logits = self.model(batch)
                logits_list.append(logits)
                labels_list.append(label)

            logits = torch.cat(logits_list).to(self.device)
            labels = torch.cat(labels_list).to(self.device)

        # Calculate FL and ECE before temperature scaling
        before_temperature_fl = fl_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        print('Before temperature - FL: %.3f, ECE: %.3f' % (before_temperature_fl, before_temperature_ece))

        # Next: optimize the temperature w.r.t. FL
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def evaluate():
            loss = fl_criterion(self.scale_temperature(logits), labels)
            loss.backward()

            return loss

        optimizer.step(evaluate)

        # Calculate FL and ECE after temperature scaling
        after_temperature_fl = fl_criterion(self.scale_temperature(logits), labels).item()
        after_temperature_ece = ece_criterion(self.scale_temperature(logits), labels).item()
        print('Optimal temperature: %.3f' % self.temperature.item())
        print('After temperature - FL: %.3f, ECE: %.3f' % (after_temperature_fl, after_temperature_ece))

        return self

class AlignmentTemperatureScaler(TemperatureScaler):
    """
    TemperatureScaler for reference models.
    """
    def __init__(self, model, device='cpu'):
        super(AlignmentTemperatureScaler, self).__init__(model, device)

    def forward(self, batch, template_batch, template_labels):
        logits = self.model(batch, template_batch, template_labels)

        return self.sigmoid(self.scale_temperature(logits))

    def set_temperature(
        self, val_loader, val_templates_loader, alpha=0.25, gamma=2):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize FL.
        val_loader (DataLoader): validation set loader
        """
        self.to(self.device)
        fl_criterion = FocalLossBinary(
            alpha=alpha,
            gamma=gamma,
            logits=True
        ).to(self.device)
        ece_criterion = ECELossBinary(logits=True).to(self.device)

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for batch, label in val_loader:
                templates, template_labels = next(val_templates_loader)
                batch = batch.to(self.device)
                templates = templates.to(self.device)
                template_labels = template_labels.to(self.device)
                logits = self.model(batch, templates, template_labels)
                logits_list.append(logits)
                labels_list.append(label)

            logits = torch.cat(logits_list).to(self.device)
            labels = torch.cat(labels_list).to(self.device)

        # Calculate FL and ECE before temperature scaling
        before_temperature_fl = fl_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        print('Before temperature - FL: %.3f, ECE: %.3f' % (before_temperature_fl, before_temperature_ece))

        # Next: optimize the temperature w.r.t. FL
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def evaluate():
            loss = fl_criterion(self.scale_temperature(logits), labels)
            loss.backward()

            return loss

        optimizer.step(evaluate)

        # Calculate FL and ECE after temperature scaling
        after_temperature_fl = fl_criterion(self.scale_temperature(logits), labels).item()
        after_temperature_ece = ece_criterion(self.scale_temperature(logits), labels).item()
        print('Optimal temperature: %.3f' % self.temperature.item())
        print('After temperature - FL: %.3f, ECE: %.3f' % (after_temperature_fl, after_temperature_ece))

        return self
