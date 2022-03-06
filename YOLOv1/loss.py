import torch
import torch.nn as nn
from utils import intersection_over_union


class YOLOLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super(YOLOLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.S = S
        self.B = B
        self.C = C
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)

        iou_bbox1 = intersection_over_union(predictions[..., 21:25], target[..., 21:25])
        iou_bbox2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25])

        ious = torch.cat([iou_bbox1.unsqueeze(0), iou_bbox2.unsqueeze(0)], dim=0)
        iou_max, best_box = torch.max(ious, dim=0)

        """
        --- Identity of object i (Iobj_i)
        Basically it verifies whether there is an object inside the cell i based on its confidence score.
        """
        exists_box = target[..., 20].unsqueeze(3)

        # =======================
        #   FOR BOX COORDINATES
        # =======================
        box_predictions = exists_box * (
                best_box * predictions[..., 26:30]
                + (1 - best_box) * predictions[..., 21:25]
        )
        box_targets = exists_box * target[..., 21:25]

        """
        * Why torch.abs? because at the initial part of learning, some predictions could be negative (because of 
        LeakyReLU), so we take abs to avoid causing error.
        * Why torch.sign? because we still want to have the negative value for the bounding boxes with negative width
        and height.
          
        """
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6)
        )
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        box_loss = self.mse(box_predictions, box_targets)  # box loss is a number

        # ====================
        #   FOR OBJECT LOSS
        # ====================
        pred_box = (
                best_box * predictions[..., 25:26] + (1 - best_box) * predictions[..., 20:21]
        )
        object_loss = self.mse(
            exists_box * pred_box,
            exists_box * target[..., 20:21]
        )

        # =======================
        #   FOR NO OBJECT LOSS
        # =======================
        no_object_loss = self.mse(
            (1 - exists_box) * predictions[..., 20:21],
            (1 - exists_box) * target[..., 20:21]
        )

        no_object_loss += self.mse(
            (1 - exists_box) * predictions[..., 25:26],
            (1 - exists_box) * target[..., 20:21]
        )

        # ==================
        #   FOR CLASS LOSS
        # ==================
        class_loss = self.mse(
            exists_box * predictions[..., :20],
            exists_box * target[..., 20]
        )

        loss = self.lambda_coord * box_loss \
               + object_loss \
               + self.lambda_noobj * no_object_loss \
               + class_loss

        return loss
