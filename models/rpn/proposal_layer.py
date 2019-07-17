import numpy as np
import torch

def bbox_transform_inv_1d(boxes, deltas):
    if len(boxes) == 0:
        return deltas.detach() * 0

    batch_size = deltas.size(0)
    boxes_dims = boxes.size()

    boxes = boxes.unsqueeze(0).expand(batch_size, *boxes_dims)

    widths = boxes[:, :, 1] - boxes[:, :, 0] + 1.0
    ctr_x = boxes[:, :, 0] + 0.5 * widths

    dx = deltas[:, :, 0]
    dw = deltas[:, :, 1]

    pred_ctr_x = dx.unsqueeze(2) * widths.float().unsqueeze(2) + ctr_x.float().unsqueeze(2)
    pred_w = torch.exp(dw).unsqueeze(2) * widths.float().unsqueeze(2)

    pred_boxes = deltas.clone()

    pred_boxes[:, :, 0::2] = pred_ctr_x - 0.5 * pred_w
    pred_boxes[:, :, 1::2] = pred_ctr_x + 0.5 * pred_w

    return pred_boxes

def clip_boxes_1d(boxes, seq_len):
    boxes[:, :, 0] = boxes[:, :, 0].clamp(0, seq_len - 1)
    boxes[:, :, 1] = boxes[:, :, 1].clamp(0, seq_len - 1)

    return boxes

def compute_ious_1d(proposal, proposal_width, proposals, proposal_widths):
    lefts = torch.max(proposal[0], proposals[:, 0])
    rights = torch.min(proposal[1], proposals[:, 1])

    intersections = (rights - lefts + 1).clamp(min=0)

    unions = proposal_width + proposal_widths - intersections

    ious = intersections / unions

    return ious

def non_maximum_suppression_1d(proposals, scores, threshold):
    left = proposals[:, 0]
    right = proposals[:, 1]

    widths = right - left + 1.0

    scores, order = scores.view(-1).sort(descending=True)

    keep = []
    while order.size(0) > 0:
        idx = order[0]
        keep.append(idx)
        order = order[order != idx]

        if order.size(0) == 0:
            break

        ious = compute_ious_1d(
            proposals[idx], widths[idx], proposals[order], widths[order])

        selected_idxs = (ious <= threshold)

        order = torch.masked_select(order, selected_idxs)

    if type(keep[0]) != list:
        keep = [element.item() for element in keep]

    return torch.from_numpy(np.array(keep))

def proposal_layer_1d(
    rpn_cls_prob,
    rpn_bbox_pred,
    seq_len,
    anchors,
    num_anchors,
    pre_nms_topN,
    nms_threshold,
    post_nms_topN):
    batch_size = rpn_cls_prob.size(0)

    scores = rpn_cls_prob.view(batch_size, -1)
    rpn_bbox_pred = rpn_bbox_pred.view(batch_size, -1, 2)

    proposals = bbox_transform_inv_1d(anchors, rpn_bbox_pred)

    proposals = clip_boxes_1d(proposals, seq_len)

    _, order = torch.sort(scores, dim=1, descending=True)

    output = torch.tensor(()).new_zeros(batch_size, post_nms_topN, 4)

    for i in range(batch_size):
        proposals_i = proposals[i]
        scores_i = scores[i]

        order_i = order[i]

        if pre_nms_topN > 0:
            order_i = order_i[:pre_nms_topN]

        proposals_i = proposals_i[order_i, :]
        scores_i = scores_i[order_i].view(-1, 1)

        keep = non_maximum_suppression_1d(
            proposals_i, scores_i, nms_threshold)

        if post_nms_topN > 0:
            keep = keep[:post_nms_topN]
            
        proposals_i = proposals_i[keep, :]
        scores_i = scores_i[keep, :]

        num_proposals_i = proposals_i.size(0)

        output[i, :, 0] = i
        output[i, :num_proposals_i, 1:3] = proposals_i
        output[i, :num_proposals_i, 3] = scores_i.view(-1)

    return output
