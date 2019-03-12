import random
import torch

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from torch.utils.data import DataLoader, Subset

def get_data_loaders(
    data, test_batch_proportion=0.1, train_batch_size=1, val_batch_size=1):
    n = len(data)
    n_test = int(n * test_batch_proportion)
    n_train = n - 2 * n_test

    idx = list(range(n))
    random.shuffle(idx)

    train_idx = idx[:n_train]
    val_idx = idx[n_train:(n_train + n_test)]
    test_idx = idx[(n_train + n_test):]

    train_set = Subset(data, train_idx)
    val_set = Subset(data, val_idx)
    test_set = Subset(data, test_idx)

    train_loader = DataLoader(train_set, batch_size=train_batch_size)
    val_loader = DataLoader(val_set, batch_size=val_batch_size)
    test_loader = DataLoader(test_set, batch_size=val_batch_size)

    return train_loader, val_loader, test_loader

def train(data, model, optimizer=None, loss=None, device='cpu', **kwargs):
    train_loader, val_loader, test_loader = get_data_loaders(
        data, kwargs['test_batch_proportion'],
        kwargs['train_batch_size'],
        kwargs['val_batch_size'])

    if not optimizer:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    if not loss:
        loss = torch.nn.BCELoss()

    trainer = create_supervised_trainer(model, optimizer, loss)
    evaluator = create_supervised_evaluator(model,
                                            metrics={
                                                'accuracy': Accuracy(),
                                                'loss': Loss(loss)
                                            })

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(trainer):
        print("Epoch[{}] Loss: {:.4f}".format(
            trainer.state.epoch, trainer.state.output))

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        print("Training Results - Epoch: {} \
               Avg accuracy: {:.4f} Avg loss: {:.4f}"
               .format(
                   trainer.state.epoch, metrics['accuracy'], metrics['loss']))

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        print("Validation Results - Epoch: {} \
               Avg accuracy: {:.4f} Avg loss: {:.4f}"
               .format(
                   trainer.state.epoch, metrics['accuracy'], metrics['loss']))

    @trainer.on(Events.COMPLETED)
    def log_test_results(trainer):
        evaluator.run(test_loader)
        metrics = evaluator.state.metrics
        print("Test Results - Avg accuracy: {:.4f} Avg loss: {:.4f}"
            .format(metrics['accuracy'], metrics['loss']))

    trainer.run(train_loader, max_epochs=kwargs['max_epochs'])
