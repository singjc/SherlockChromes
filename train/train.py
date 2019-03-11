import torch

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator

def train(data, model, optimizer=None, loss=None, **kwargs):
    train_loader, val_loader = get_data_loaders(
        kwargs['train_batch_size'], kwargs['val_batch_size'])

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
        print("Validation Results - Epoch: {} \
               Avg accuracy: {:.4f} Avg loss: {:.4f}"
              .format(
                  trainer.state.epoch, metrics['accuracy'], metrics['loss']))

    trainer.run(train_loader, max_epochs=kwargs['max_epochs'])
