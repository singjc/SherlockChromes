import importlib
import numpy as np
import os
import random
import torch

from ignite.engine import Engine, Events
from ignite.handlers import EarlyStopping, ModelCheckpoint
from ignite.metrics import Accuracy, Loss, Precision, Recall
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Subset


def get_data_loaders(
    data,
    test_batch_proportion=0.1,
    batch_size=1,
    sampling_fn=None,
    collate_fn=None,
    outdir_path=None
):

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

    if outdir_path:
        if not os.path.isdir(outdir_path):
            os.mkdir(outdir_path)

        np.savetxt(os.path.join(outdir_path, 'train_idx.txt'), np.array(
            train_idx))
        np.savetxt(os.path.join(outdir_path, 'val_idx.txt'), np.array(
            val_idx))
        np.savetxt(os.path.join(outdir_path, 'test_idx.txt'), np.array(
            test_idx))

    train_set = Subset(data, train_idx)
    val_set = Subset(data, val_idx)
    test_set = Subset(data, test_idx)

    if collate_fn:
        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            collate_fn=collate_fn)
        val_loader = DataLoader(
            val_set,
            batch_size=batch_size,
            collate_fn=collate_fn)
        test_loader = DataLoader(
            test_set,
            batch_size=batch_size,
            collate_fn=collate_fn)
    else:
        train_loader = DataLoader(
            train_set,
            batch_size=batch_size)
        val_loader = DataLoader(
            val_set,
            batch_size=batch_size)
        test_loader = DataLoader(
            test_set,
            batch_size=batch_size)

    return train_loader, val_loader, test_loader


def train(
    data,
    model,
    optimizer=None,
    loss=None,
    sampling_fn=None,
    collate_fn=None,
    device='cpu',
    **kwargs
):
    train_loader, val_loader, test_loader = get_data_loaders(
        data, kwargs['test_batch_proportion'],
        kwargs['batch_size'],
        sampling_fn,
        collate_fn,
        kwargs['outdir_path'])

    use_visdom = False

    if 'visualize' in kwargs and kwargs['visualize']:
        visdom_spec = importlib.util.find_spec('visdom')
        visdom_available = visdom_spec is not None

        if visdom_available:
            print('Visdom detected!')
            import visdom
            from visualizations import Visualizations

            vis = Visualizations(env_name=kwargs['model_savename'])
            use_visdom = True

    if not optimizer:
        optimizer = torch.optim.Adam(model.parameters())

    if not loss:
        loss = torch.nn.BCELoss()

    if 'transfer_model_path' in kwargs:
        model.load_state_dict(
            torch.load(kwargs['transfer_model_path']).state_dict(),
            strict=False
        )

    model.to(device)

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        x, y = batch
        x = x.to(device=device, non_blocking=False)
        y = np.array([y])
        print('Train Bbox Label: ' + str(y[0]))
        y_pred = model(x, y)
        loss_out = loss(y_pred, y)
        loss_out.backward()
        optimizer.step()
        return loss_out.item()

    trainer = Engine(_update)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            x, y = batch
            x = x.to(device=device, non_blocking=False)
            y = np.array([y])
            print('Eval Bbox Label: ' + str(y[0]))
            y_pred = model(x, y)
            loss_out = loss(y_pred, y)
            return loss_out.item()

    evaluator = Engine(_inference)

    scheduler = CosineAnnealingWarmRestarts(
        optimizer, kwargs['T_0'], T_mult=kwargs['T_mult'])

    def step_scheduler(engine, scheduler):
        scheduler.step()

    trainer.add_event_handler(
        Events.EPOCH_STARTED, step_scheduler, scheduler)

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(trainer):
        print("Epoch[{}] Loss: {:.8f}".format(
            trainer.state.epoch, trainer.state.output))

    def neg_loss(engine):
        loss = engine.state.output

        return -loss

    checkpoint_score = ModelCheckpoint(
        kwargs['outdir_path'],
        kwargs['model_savename'],
        score_function=neg_loss,
        score_name='loss',
        n_saved=5)

    checkpoint_interval = ModelCheckpoint(
        kwargs['outdir_path'],
        kwargs['model_savename'],
        save_interval=5,
        n_saved=5)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        print("Training Results - Epoch: {} Avg loss: {:.8f}"
               .format(
                   trainer.state.epoch,
                   evaluator.state.output))
        if use_visdom:
            vis.plot_train_loss(evaluator.state.output, trainer.state.epoch)

        if kwargs['test_batch_proportion'] == 0.0:
            checkpoint_score(evaluator, {'model': model})
            checkpoint_interval(evaluator, {'model': model})

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        if kwargs['test_batch_proportion'] == 0.0:
            return

        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        print("Validation Results - Epoch: {} Avg loss: {:.8f}"
            .format(
                trainer.state.epoch,
                evaluator.state.output))
        if use_visdom:
            vis.plot_val_loss(evaluator.state.output, trainer.state.epoch)
        checkpoint_score(evaluator, {'model': model})
        checkpoint_interval(evaluator, {'model': model})

    @trainer.on(Events.COMPLETED)
    def log_test_results(trainer):
        if kwargs['test_batch_proportion'] == 0.0:
            return
   
        evaluator.run(test_loader)
        metrics = evaluator.state.metrics
        print("Test Results - Epoch: {} Avg loss: {:.8f}"
            .format(
                trainer.state.epoch,
                evaluator.state.output))

    trainer.run(train_loader, max_epochs=kwargs['max_epochs'])
