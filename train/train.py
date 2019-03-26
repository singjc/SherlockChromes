import random
import sys
import torch

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import EarlyStopping, ModelCheckpoint
from ignite.metrics import Accuracy, Loss, Precision, Recall
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset

def get_data_loaders(
    data,
    test_batch_proportion=0.1,
    batch_size=1,
    sampling_fn=None,
    collate_fn=None):

    if sampling_fn:
        train_idx, val_idx, test_idx = sampling_fn(
            data, test_batch_proportion)
    else:
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

    if collate_fn:
        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            collate_fn=collate_fn,
            drop_last=True)
        val_loader = DataLoader(
            val_set,
            batch_size=batch_size,
            collate_fn=collate_fn,
            drop_last=True)
        test_loader = DataLoader(
            test_set,
            batch_size=batch_size,
            collate_fn=collate_fn,
            drop_last=True)
    else:
        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            drop_last=True)
        val_loader = DataLoader(
            val_set,
            batch_size=batch_size,
            drop_last=True)
        test_loader = DataLoader(
            test_set,
            batch_size=batch_size,
            drop_last=True)

    return train_loader, val_loader, test_loader

def train(
    data,
    model,
    optimizer=None,
    loss=None,
    sampling_fn=None,
    collate_fn=None,
    device='cpu',
    **kwargs):
    train_loader, val_loader, test_loader = get_data_loaders(
        data, kwargs['test_batch_proportion'],
        kwargs['batch_size'],
        sampling_fn,
        collate_fn)

    if not optimizer:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    if not loss:
        loss = torch.nn.BCELoss()

    trainer = create_supervised_trainer(model, optimizer, loss, device=device)
    evaluator = create_supervised_evaluator(model,
                                            metrics={
                                                'accuracy': Accuracy(),
                                                'precision': Precision(),
                                                'recall': Recall(),
                                                'loss': Loss(loss)
                                            },
                                            device=device)

    scheduler = CosineAnnealingLR(
        optimizer, kwargs['lr_cycle_len'], kwargs['lr_min'])
    
    def cosine_annealing_scheduler(engine, scheduler):
        scheduler.step()

    trainer.add_event_handler(
        Events.EPOCH_STARTED, cosine_annealing_scheduler, scheduler)

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(trainer):
        print("Epoch[{}] Loss: {:.4f}".format(
            trainer.state.epoch, trainer.state.output))

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        print("Training Results - Epoch: {} Avg accuracy: {:.4f} Avg precision: {:.4f} Avg recall: {:.4f} Avg loss: {:.4f}"
               .format(
                   trainer.state.epoch,
                   metrics['accuracy'],
                   metrics['precision'],
                   metrics['recall'],
                   metrics['loss']))

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        if kwargs['mode'] != 'train only':
            evaluator.run(val_loader)
            metrics = evaluator.state.metrics
            print("Validation Results - Epoch: {} Avg accuracy: {:.4f} Avg precision: {:.4f} Avg recall: {:.4f} Avg loss: {:.4f}"
                .format(
                    trainer.state.epoch,
                    metrics['accuracy'],
                    metrics['precision'],
                    metrics['recall'],
                    metrics['loss']))

    def f1(engine):
        metrics = engine.state.metrics
        precision = metrics['precision']
        recall = metrics['recall']
        f1 = 2 * (precision * recall) / (precision + recall)

        return f1

    checkpoint = ModelCheckpoint(
        kwargs['model_savedir'],
        kwargs['model_savename'],
        score_function=f1,
        score_name='f1')

    evaluator.add_event_handler(Events.COMPLETED, checkpoint, {'model': model})

    # TODO: Uncomment once early stopping plateau fixed released
    # def val_loss(engine):
    #     val_loss = float('{:.4f}'.format(engine.state.metrics['loss']))

    #     return val_loss

    # early_stopping = EarlyStopping(
    #     patience=2, score_function=val_loss, trainer=trainer)

    # evaluator.add_event_handler(Events.COMPLETED, early_stopping)

    @trainer.on(Events.COMPLETED)
    def log_test_results(trainer):
        if kwargs['mode'] != 'train only':
            evaluator.run(test_loader)
            metrics = evaluator.state.metrics
            print("Test Results - Epoch: {} Avg accuracy: {:.4f} Avg precision: {:.4f} Avg recall: {:.4f} Avg loss: {:.4f}"
                .format(
                    trainer.state.epoch,
                    metrics['accuracy'],
                    metrics['precision'],
                    metrics['recall'],
                    metrics['loss']))

    trainer.run(train_loader, max_epochs=kwargs['max_epochs'])
