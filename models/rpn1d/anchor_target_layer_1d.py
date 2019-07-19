import numpy as np
import torch

def bbox_overlaps_1d(boxes, query_boxes):
    if isinstance(boxes, np.ndarray):
        boxes = torch.from_numpy(boxes)
        query_boxes = torch.from_numpy(query_boxes)
        out_fn = lambda x: x.numpy()
    else:
        out_fn = lambda x: x

    box_widths = boxes[:, 1] - boxes[:, 0] + 1.0
    query_box_widths = query_boxes[:, 1] - query_boxes[:, 0] + 1.0

    lefts = torch.max(boxes[:, 0:1], query_boxes[:, 0:1].t())
    rights = torch.min(boxes[:, 1:2], query_boxes[:, 1:2].t())

    intersections = (rights - lefts + 1).clamp(min=0)

    unions = box_widths.view(-1, 1) + query_box_widths.view(1, -1) - intersections

    overlaps = intersections / unions

    return overlaps

def bbox_transform_1d(ex_rois, gt_rois):
    ex_widths = ex_rois[:, 1] - ex_rois[:, 0] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths

    gt_widths = gt_rois[:, 1] - gt_rois[:, 0] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dw = torch.log(gt_widths.float() / ex_widths.float())

    targets = torch.stack((targets_dx.float(), targets_dw), 1)

    return targets

def unmap(data, count, idxs, fill=0):
    if len(data.shape) == 1:
        ret = np.empty((count, ), dtype=np.float32)
        ret.fill(fill)
        ret[idxs] = data
    else:
        ret = np.empty((count, ) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[idxs, :] = data

    return ret

def anchor_target_layer_1d(
    gt_boxes,
    anchors,
    num_anchors,
    seq_len,
    overwrite_positives=False,
    negative_overlap=0.3,
    positive_overlap=0.7,
    fg_fraction=0.5,
    rpn_batchsize=256):
    A = num_anchors
    total_anchors = anchors.shape[0]
    K = total_anchors / num_anchors

    allowed_border = 0

    anchors = anchors.cpu()
    
    idxs_inside = np.where(
        (anchors[:, 0] >= -allowed_border) &
        (anchors[:, 1] < seq_len + allowed_border)
    )[0]

    anchors = anchors[idxs_inside, :]

    labels = np.empty((len(idxs_inside), ), dtype=np.float32)
    labels.fill(-1)

    overlaps = bbox_overlaps_1d(
        np.ascontiguousarray(anchors, dtype=np.float),
        np.ascontiguousarray(gt_boxes, dtype=np.float)
    )

    if not overwrite_positives:
        labels[np.where(overlaps < negative_overlap)[0]] = 0

    labels[np.where(overlaps >= positive_overlap)[0]] = 1

    if overwrite_positives:
        labels[np.where(overlaps < negative_overlap)[0]] = 0

    num_fg = int(fg_fraction * rpn_batchsize)
    fg_idxs = np.where(labels == 1)[0]

    if len(fg_idxs) > num_fg:
        disable_idxs = np.random.choice(
            fg_idxs, size=(len(fg_idxs) - num_fg),
            replace=False
        )
        labels[disable_idxs] = -1

    num_bg = int(rpn_batchsize) - np.sum(labels == 1)
    bg_idxs = np.where(labels == 0)[0]

    if len(bg_idxs) > num_bg:
        disable_idxs = np.random.choice(
            bg_idxs, size=(len(bg_idxs) - num_bg),
            replace=False
        )
        labels[disable_idxs] = -1

    bbox_targets = np.zeros((len(idxs_inside), 2), dtype=np.float32)
    bbox_targets = bbox_transform_1d(
        anchors,
        torch.from_numpy(gt_boxes)).numpy()

    bbox_inside_weights = np.zeros((len(idxs_inside), 2), dtype=np.float32)
    bbox_inside_weights[labels == 1, :] = np.array((1.0, 1.0))

    bbox_outside_weights = np.zeros((len(idxs_inside), 2), dtype=np.float32)

    num_examples = np.sum(labels >= 0)
    positive_weights = np.ones((1, 2)) * 1.0 / num_examples
    negative_weights = np.ones((1, 2)) * 1.0 / num_examples

    bbox_outside_weights[labels == 1, :] = positive_weights
    bbox_outside_weights[labels == 0, :] = negative_weights

    labels = unmap(labels, total_anchors, idxs_inside, fill=-1)
    bbox_targets = unmap(bbox_targets, total_anchors, idxs_inside, fill=0)

    bbox_inside_weights = unmap(
        bbox_inside_weights, total_anchors, idxs_inside, fill=0
    )
    bbox_outside_weights = unmap(
        bbox_outside_weights, total_anchors, idxs_inside, fill=0
    )

    labels = labels.reshape((1, seq_len, A)).transpose(0, 2, 1)
    labels = labels.reshape((1, 1, A * seq_len))

    bbox_targets = bbox_targets.reshape((1, seq_len, A * 2))

    bbox_inside_weights = bbox_inside_weights.reshape((1, seq_len, A * 2))

    bbox_outside_weights = bbox_outside_weights.reshape((1, seq_len, A * 2))

    return labels, bbox_targets, bbox_inside_weights, bbox_outside_weights
