import torch

from IoU import intersection_over_union


def non_max_suppression(bounding_boxes, iou_threshold, probability_threshold, box_format='corners'):
    # bounding_boxes = [[class_id, probability_of_bounding_box, x1, y1, x2, y2], [...], [...]]

    assert type(bounding_boxes) == list

    bounding_boxes = [box for box in bounding_boxes if box[1] > probability_threshold]
    bounding_boxes_after_nms = []
    bounding_boxes = sorted(bounding_boxes, key=lambda x: x[1], reverse=True)

    while bounding_boxes:
        chosen_box = bounding_boxes.pop(0)

        bounding_boxes = [
            box
            for box in bounding_boxes
            if box[0] != chosen_box[0]
               or intersection_over_union(torch.tensor(chosen_box[2:]),
                                          torch.tensor(box[2:]),
                                          box_format) < iou_threshold
        ]

        bounding_boxes_after_nms.append(chosen_box)

    return bounding_boxes_after_nms
