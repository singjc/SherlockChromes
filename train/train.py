import random
import torch

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from torch.utils.data import DataLoader, Subset

def get_data_loaders(data, test_batch_proportion):
    n = len(data)
    n_test = int(n * test_batch_proportion)
    n_train = n - 2 * n_test

    idx = list(range(n))
    random.shuffle(idx)

    train_idx = idx[:n_train]
    val_idx = idx[n_train:(n_train + n_test)]
    test_idx = idx[(n_train + n_test):]

    train_set = data.Subset(data, train_idx)
    val_set = data.Subset(data, val_idx)
    test_set = data.Subset(data, test_idx)

    train_loader = DataLoader(train_set)
    val_loader = DataLoader(val_set)
    test_loader = DataLoader(test_set)

    return train_loader, val_loader, test_loader

def train(data, model, optimizer=None, loss=None, **kwargs):
    train_loader, val_loader, test_loader = get_data_loaders(
        data, kwargs['test_batch_proportion'])

    if not optimizer:
        optimizer = torch.optim.Adam(model.parameters(), lr=kwargs['lr'])

    if not loss:
        loss = torch.nn.NLLLoss()

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
