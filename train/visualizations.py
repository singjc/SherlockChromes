import visdom

from datetime import datetime

class Visualizations:
    def __init__(self, env_name=None):
        if env_name is None:
            env_name = str(datetime.now().strftime('%d-%m %Hh%M'))
        
        self.env_name = env_name
        self.vis = visdom.Visdom(
            env=self.env_name,
            log_to_filename='../../../data/logs/' + self.env_name + '.log')
        self.train_acc_win = 'train_acc'
        self.val_acc_win = 'val_acc'
        self.train_prec_win = 'train_prec'
        self.val_prec_win = 'val_prec'
        self.train_recall_win = 'train_recall'
        self.val_recall_win = 'val_recall'
        self.train_loss_win = 'train_loss'
        self.val_loss_win = 'val_loss'

    def plot_vis_line(self, value, step, win, xlabel, ylabel, title):
        return self.vis.line(
            [value],
            [step],
            win=win,
            update='append' if win else None,
            opts=dict(
                xlabel=xlabel,
                ylabel=ylabel,
                title=title
            )
        )

    def plot_train_acc(self, acc, epoch):
        self.train_acc_win = self.plot_vis_line(
            acc,
            epoch,
            self.train_acc_win,
            'Epoch',
            'Accuracy',
            'Train Accuracy'
        )
    
    def plot_val_acc(self, acc, epoch):
        self.val_acc_win = self.plot_vis_line(
            acc,
            epoch,
            self.val_acc_win,
            'Epoch',
            'Accuracy',
            'Val Accuracy'
        )

    def plot_train_prec(self, prec, epoch):
        self.train_prec_win = self.plot_vis_line(
            prec,
            epoch,
            self.train_prec_win,
            'Epoch',
            'Precision',
            'Train Precision'
        )
    
    def plot_val_prec(self, prec, epoch):
        self.val_prec_win = self.plot_vis_line(
            prec,
            epoch,
            self.val_prec_win,
            'Epoch',
            'Precision',
            'Val Precision'
        )

    def plot_train_recall(self, recall, epoch):
        self.train_recall_win = self.plot_vis_line(
            recall,
            epoch,
            self.train_recall_win,
            'Epoch',
            'Recall',
            'Train Recall'
        )
    
    def plot_val_recall(self, recall, epoch):
        self.val_recall_win = self.plot_vis_line(
            recall,
            epoch,
            self.val_recall_win,
            'Epoch',
            'Recall',
            'Val Recall'
        )

    def plot_train_loss(self, loss, epoch):
        self.train_loss_win = self.plot_vis_line(
            loss,
            epoch,
            self.train_loss_win,
            'Epoch',
            'Loss',
            'Train Loss'
        )
    
    def plot_val_loss(self, loss, epoch):
        self.val_loss_win = self.plot_vis_line(
            loss,
            epoch,
            self.val_loss_win,
            'Epoch',
            'Loss',
            'Val Loss'
        )
