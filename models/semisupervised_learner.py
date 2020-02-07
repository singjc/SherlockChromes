import copy
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '../optimizers')

from focal_loss import FocalLossBinary

class SemiSupervisedLearner(nn.Module):
    def __init__(
        self,
        model,
        wu=1,
        threshold=0.95,
        weak_p=0.1,
        weak_drop_channels=False,
        strong_p=0.3,
        strong_drop_channels=False,
        loss_alpha=0.25,
        loss_gamma=2,
        loss_logits=False,
        loss_reduction='none',
        debug=False):
        super(SemiSupervisedLearner, self).__init__()
        self.segmentator = copy.deepcopy(model)
        self.wu = wu
        self.threshold = threshold

        if weak_drop_channels:
            self.weak_augmentator = nn.Dropout2d(weak_p)
        else:
            self.weak_augmentator = nn.Dropout(weak_p)
        
        if strong_drop_channels:
            self.strong_augmentator = nn.Dropout2d(strong_p)
        else:
            self.strong_augmentator = nn.Dropout(strong_p)

        self.loss = FocalLossBinary(
            loss_alpha, loss_gamma, loss_logits, loss_reduction
        )

        self.debug = debug

    def forward(self, unlabeled_batch, labeled_batch=None, labels=None):
        if self.training:
            assert labeled_batch is not None, 'missing labeled data!'
            assert labels is not None, 'missing labels!'
            labeled_loss = torch.mean(
                self.loss(self.segmentator(labeled_batch), labels)
            )

            strongly_augmented = self.strong_augmentator(unlabeled_batch)
            weakly_augmented = self.weak_augmentator(unlabeled_batch)
            weak_output = self.segmentator(weakly_augmented)
            pseudo_labels = (weak_output >= self.threshold).float()
            quality_mask =  pseudo_labels.reshape(1, -1).squeeze()

            unlabeled_loss = torch.mean(
                quality_mask * 
                self.loss(self.segmentator(strongly_augmented), pseudo_labels)
            )
            
            if self.debug:
                print(
                    f'Labeled Loss: {labeled_loss.item()}, '
                    f'Unlabeled Loss: {unlabeled_loss.item()}, '
                    f'Weighted UL Loss: {self.wu * unlabeled_loss.item()}'
                )

            return labeled_loss + self.wu * unlabeled_loss
        else:
            return self.segmentator(unlabeled_batch)

class SemiSupervisedAlignmentLearner(SemiSupervisedLearner):
    def __init__(
        self,
        model,
        wu=1,
        threshold=0.85,
        weak_p=0.2,
        weak_drop_channels=True,
        strong_p=0.3,
        strong_drop_channels=True,
        loss_alpha=0.25,
        loss_gamma=2,
        loss_logits=False,
        loss_reduction='none',
        debug=False):
        super(SemiSupervisedAlignmentLearner, self).__init__(
            model,
            wu=wu,
            threshold=threshold,
            weak_p=weak_p,
            weak_drop_channels=weak_drop_channels,
            strong_p=strong_p,
            strong_drop_channels=strong_drop_channels,
            loss_alpha=loss_alpha,
            loss_gamma=loss_gamma,
            loss_logits=loss_logits,
            loss_reduction=loss_reduction,
            debug=debug
        )

    def forward(
        self,
        unlabeled_batch,
        template,
        template_label,
        labeled_batch=None,
        labels=None):
        if self.training:
            assert labeled_batch is not None, 'missing labeled data!'
            assert labels is not None, 'missing labels!'
            labeled_loss = torch.mean(
                self.loss(
                    self.segmentator(labeled_batch, template, template_label),
                    labels
                )
            )

            strongly_augmented = self.strong_augmentator(unlabeled_batch)
            weakly_augmented = self.weak_augmentator(unlabeled_batch)
            weak_output = self.segmentator(weakly_augmented, template, template_label)
            pseudo_labels = (weak_output >= self.threshold).float()
            quality_mask =  pseudo_labels.reshape(1, -1).squeeze()

            unlabeled_loss = torch.mean(
                quality_mask * 
                self.loss(
                    self.segmentator(
                        strongly_augmented, template, template_label),
                    pseudo_labels
                )
            )
            
            if self.debug:
                print(
                    f'Labeled Loss: {labeled_loss.item()}, '
                    f'Unlabeled Loss: {unlabeled_loss.item()}, '
                    f'Weighted UL Loss: {self.wu * unlabeled_loss.item()}'
                )

            return labeled_loss + self.wu * unlabeled_loss
        else:
            return self.segmentator(unlabeled_batch, template, template_label)
