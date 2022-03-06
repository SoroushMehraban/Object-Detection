import torch
from collections import Counter
from IoU import intersection_over_union


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

