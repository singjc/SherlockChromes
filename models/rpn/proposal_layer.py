import numpy as np
import torch

def bbox_transform_inv_1d(boxes, deltas):
    if len(boxes) == 0:
        return deltas.detach() * 0

    widths = boxes[:, 1] - boxes[:, 0] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths

    dx = deltas[:, 0]
    dw = deltas[:, 1]

    pred_ctr_x = dx.unsqueeze(1) * widths.float().unsqueeze(1) + ctr_x.float().unsqueeze(1)
    pred_w = torch.exp(dw).unsqueeze(1) * widths.float().unsqueeze(1)

    pred_boxes = torch.cat(
        [
            pred_ctr_x - 0.5 * pred_w,
            pred_ctr_x + 0.5 * pred_w
        ],
        dim=1
    )

    return pred_boxes

def clip_boxes_1d(boxes, seq_len):
    boxes = torch.stack(
        [
            boxes[:, 0].clamp(0, seq_len - 1),
            boxes[:, 1].clamp(0, seq_len - 1)
        ],
        dim=1
    )

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

    print(keep)

    if type(keep[0]) != list:
        keep = [element.item() for element in keep]

    print(keep)

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
    scores = rpn_cls_prob.view(-1, 1)
    rpn_bbox_pred = rpn_bbox_pred.view(-1, 2)

    proposals = bbox_transform_inv_1d(anchors, rpn_bbox_pred)

    proposals = clip_boxes_1d(proposals, seq_len)

    scores, order = scores.view(-1).sort(descending=True)

    if pre_nms_topN > 0:
        order = order[:pre_nms_topN]
        scores = scores[:pre_nms_topN].view(-1, 1)

    proposals = proposals[order, :]

    keep = non_maximum_suppression_1d(proposals, scores, nms_threshold)

    if post_nms_topN > 0:
        keep = keep[:post_nms_topN]
        
    proposals = proposals[keep, :]
    scores = scores[keep, :]

    batch_inds = proposals.new_zeros(proposals.size(0), 1)
    blob = torch.cat((batch_inds, proposals), 1)

    return blob, scores
