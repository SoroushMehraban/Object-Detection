import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from collections import Counter


def intersection_over_union(boxes_preds, boxes_labels, box_format='midpoint'):
    # boxes_preds shape is (N, 4) where N is the number of bounding boxes
    # boxes_labels shape is (N, 4)

    if box_format == 'midpoint':
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2

        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    elif box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]

        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]
    else:
        return -1

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # .clamp(0) is for the case when they do not intersect
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    union = box1_area + box2_area - intersection + 1e-6  # 1e-6 for numerical stability
    return intersection / union


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


def mean_average_precision(pred_boxes, true_boxes, iou_threshold=0.5, box_format='corners', num_classes=20):
    # pred_boxes and true_boxes is a list like:
    # [[train_index, class_prediction, probability_score, x1, y1, x2, y2], [...], [...]]

    average_precisions = []
    epsilon = 1e-6

    for class_id in range(num_classes):
        detections = []
        ground_truths = []

        for detection in pred_boxes:
            if detection[1] == class_id:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == class_id:
                ground_truths.append(true_box)

        # image 0 -> 3 bounding boxes (ground truth objects)
        # image 1 -> 5 bounding boxes
        # => amount_bounding_boxes = {0:3, 1:5}
        amount_bounding_boxes = Counter(ground_truth[0] for ground_truth in ground_truths)

        for key, val in amount_bounding_boxes.items():
            amount_bounding_boxes[key] = torch.zeros(val)
        # amount_boxes = {0:torch.tensor([0, 0, 0]), 1:torch.tensor([0, 0, 0, 0, 0])}

        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros(len(detections))
        FP = torch.zeros(len(detections))
        total_true_bboxes = len(ground_truths)

        if total_true_bboxes == 0:
            continue

        for detection_index, detection in enumerate(detections):
            ground_truth_image = [ground_truth for ground_truth in ground_truths if ground_truth[0] == detection[0]]

            best_iou = 0
            best_ground_truth_index = None
            for index, ground_truth in enumerate(ground_truth_image):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(ground_truth[3:]),
                    box_format
                )
                if iou > best_iou:
                    best_iou = iou
                    best_ground_truth_index = index

            if best_iou > iou_threshold:
                if amount_bounding_boxes[detection[0]][best_ground_truth_index] == 0:
                    TP[detection_index] = 1
                    amount_bounding_boxes[detection[0]][best_ground_truth_index] = 1
                else:
                    FP[detection_index] = 1
            else:
                FP[detection_index] = 1

        # [ 1, 1, 0, 1, 0] -> [1, 2, 2, 3, 3]
        TP_cumulative_sum = torch.cumsum(TP, dim=0)
        FP_cumulative_sum = torch.cumsum(FP, dim=0)
        recalls = TP_cumulative_sum / (total_true_bboxes + epsilon)
        precisions = TP_cumulative_sum / (TP_cumulative_sum + FP_cumulative_sum + epsilon)
        precisions = torch.cat((torch.tensor([1]), precisions))  # point (0, 1)
        recalls = torch.cat((torch.tensor([0]), recalls))
        average_precisions.append(torch.trapz(precisions, recalls))  # estimate the integral

    return sum(average_precisions) / len(average_precisions)


# ===================
# The following functions are copied from this repository:
# https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/object_detection/YOLO/utils.py
# ===================

def plot_image(image, boxes):
    """Plots predicted bounding boxes on the image"""
    im = np.array(image)
    height, width, _ = im.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)

    # box[0] is x midpoint, box[2] is width
    # box[1] is y midpoint, box[3] is height

    # Create a Rectangle potch
    for box in boxes:
        box = box[2:]
        assert len(box) == 4, "Got more values than in x, y, w, h, in a box!"
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.show()


def get_bboxes(
        loader,
        model,
        iou_threshold,
        threshold,
        pred_format="cells",
        box_format="midpoint",
        device="cuda",
):
    all_pred_boxes = []
    all_true_boxes = []

    # make sure model is in eval before get bboxes
    model.eval()
    train_idx = 0

    for batch_idx, (x, labels) in enumerate(loader):
        x = x.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]
        true_bboxes = cellboxes_to_boxes(labels)
        bboxes = cellboxes_to_boxes(predictions)

        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=iou_threshold,
                probability_threshold=threshold,
                box_format=box_format,
            )

            # if batch_idx == 0 and idx == 0:
            #    plot_image(x[idx].permute(1,2,0).to("cpu"), nms_boxes)
            #    print(nms_boxes)

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                # many will get converted to 0 pred
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

    model.train()
    return all_pred_boxes, all_true_boxes


def convert_cellboxes(predictions, S=7):
    """
    Converts bounding boxes output from Yolo with
    an image split size of S into entire image ratios
    rather than relative to cell ratios. Tried to do this
    vectorized, but this resulted in quite difficult to read
    code... Use as a black box? Or implement a more intuitive,
    using 2 for loops iterating range(S) and convert them one
    by one, resulting in a slower but more readable implementation.
    """

    predictions = predictions.to("cpu")
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, 7, 7, 30)
    bboxes1 = predictions[..., 21:25]
    bboxes2 = predictions[..., 26:30]
    scores = torch.cat(
        (predictions[..., 20].unsqueeze(0), predictions[..., 25].unsqueeze(0)), dim=0
    )
    best_box = scores.argmax(0).unsqueeze(-1)
    best_boxes = bboxes1 * (1 - best_box) + best_box * bboxes2
    cell_indices = torch.arange(7).repeat(batch_size, 7, 1).unsqueeze(-1)
    x = 1 / S * (best_boxes[..., :1] + cell_indices)
    y = 1 / S * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
    w_y = 1 / S * best_boxes[..., 2:4]
    converted_bboxes = torch.cat((x, y, w_y), dim=-1)
    predicted_class = predictions[..., :20].argmax(-1).unsqueeze(-1)
    best_confidence = torch.max(predictions[..., 20], predictions[..., 25]).unsqueeze(
        -1
    )
    converted_preds = torch.cat(
        (predicted_class, best_confidence, converted_bboxes), dim=-1
    )

    return converted_preds


def cellboxes_to_boxes(out, S=7):
    converted_pred = convert_cellboxes(out).reshape(out.shape[0], S * S, -1)
    converted_pred[..., 0] = converted_pred[..., 0].long()
    all_bboxes = []

    for ex_idx in range(out.shape[0]):
        bboxes = []

        for bbox_idx in range(S * S):
            bboxes.append([x.item() for x in converted_pred[ex_idx, bbox_idx, :]])
        all_bboxes.append(bboxes)

    return all_bboxes


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
