# coding=utf-8
# Copyright 2022 The IDEA Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
from typing import List
import torch
import torch.nn.functional as F

from detrex.modeling import SetCriterion
from detrex.utils import get_world_size, is_dist_avail_and_initialized
from detrex.layers import box_cxcywh_to_xyxy, generalized_box_iou


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs (torch.Tensor): A float tensor of arbitrary shape.
            The predictions for each example.
        targets (torch.Tensor): A float tensor with the same shape as inputs. Stores the binary
            classification label for each element in inputs
            (0 for the negative class and 1 for the positive class).
        num_boxes (int): The number of boxes.
        alpha (float, optional): Weighting factor in range (0, 1) to balance
            positive vs negative examples. Default: 0.25.
        gamma (float): Exponent of the modulating factor (1 - p_t) to
            balance easy vs hard examples. Default: 2.

    Returns:
        torch.Tensor: The computed sigmoid focal loss.
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


class RankDetrCriterion(SetCriterion):
    """This class computes the loss for Deformable-DETR
    and two-stage Deformable-DETR
    """

    def __init__(
        self,
        num_classes,
        matcher,
        weight_dict,
        losses: List[str] = ["class", "boxes"],
        eos_coef: float = 0.1,
        loss_class_type: str = "focal_loss",
        alpha: float = 0.25,
        gamma: float = 2.0,
        GIoU_aware_class_loss: bool = True,
    ):
        super(RankDetrCriterion, self).__init__(
            num_classes=num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            losses=losses,
            eos_coef=eos_coef,
            loss_class_type=loss_class_type,
            alpha=alpha,
            gamma=gamma,
        )
        self.GIoU_aware_class_loss = GIoU_aware_class_loss

    def loss_labels(self, outputs, targets, indices, num_boxes, GIoU_aware_class_loss):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o

        # Computation classification loss
        if self.loss_class_type == "ce_loss":
            loss_class = F.cross_entropy(
                src_logits.transpose(1, 2), target_classes, self.empty_weight
            )
        elif self.loss_class_type == "focal_loss":
            # src_logits: (b, num_queries, num_classes) = (2, 300, 80)
            # target_classes_one_hot = (2, 300, 80)
            target_classes_onehot = torch.zeros(
                [src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                dtype=src_logits.dtype,
                layout=src_logits.layout,
                device=src_logits.device,
            )
            target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
            target_classes_onehot = target_classes_onehot[:, :, :-1]

            if GIoU_aware_class_loss:
                # get GIoU-aware classification target: t = (GIoU + 1) / 2

                # # get normed GIoU
                bs, n_query = outputs["pred_boxes"].shape[:2]
                out_bbox = outputs["pred_boxes"].flatten(0, 1)                # tensor shape: [bs * n_query, 4]
                tgt_bbox = torch.cat([v["boxes"] for v in targets])           # tensor shape: [gt number within a batch, 4]
                bbox_giou = generalized_box_iou(
                    box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox)
                )                                                             # tensor shape: [bs * n_query, gt number within a batch]
                bbox_giou_normed = (bbox_giou + 1) / 2.0
                bbox_giou_normed = bbox_giou_normed.reshape(bs, n_query, -1)  # tensor shape: [bs, n_query, gt number within a batch]

                # # get matched gt indices: gt_indices
                for indices_idx, element in enumerate(indices):
                    if indices_idx == 0:
                        gt_indices = element[1]
                    else:
                        curr_length = gt_indices.shape[0]
                        gt_indices = torch.cat((gt_indices, element[1] + curr_length), dim=0)

                # # get the supervision with a shape of [bs, n_query, num_classes]
                class_supervision = torch.zeros(
                    [src_logits.shape[0], src_logits.shape[1]],
                    dtype=src_logits.dtype,
                    layout=src_logits.layout,
                    device=src_logits.device,
                )
                class_supervision[idx] = bbox_giou_normed[(idx[0], idx[1], gt_indices)] # idx[0]: batch idx; idx[1]: query idx; gt_indices: matched gt idx
                class_supervision = class_supervision.detach()
                target_classes_onehot_GIoU_aware = target_classes_onehot * class_supervision.unsqueeze(-1)

                # sigmoid_focal_loss supervised by target_classes_onehot_GIoU_aware
                src_prob = src_logits.sigmoid()

                # # positive samples
                bce_loss_pos = F.binary_cross_entropy_with_logits(src_logits, target_classes_onehot_GIoU_aware, reduction="none") * target_classes_onehot
                p_t_pos = torch.abs(target_classes_onehot_GIoU_aware - src_prob * target_classes_onehot) ** self.gamma

                # # negative samples
                bce_loss_neg = F.binary_cross_entropy_with_logits(src_logits, target_classes_onehot, reduction="none") * (1 - target_classes_onehot)
                p_t_neg = torch.abs(src_prob * (1 - target_classes_onehot)) ** self.gamma

                # # total loss
                loss = p_t_pos * bce_loss_pos + p_t_neg * bce_loss_neg

                if self.alpha >= 0:
                    alpha_t = self.alpha * target_classes_onehot + (1 - self.alpha) * (1 - target_classes_onehot)
                    loss = alpha_t * loss

                loss_class = loss.mean(1).sum() / num_boxes
                loss_class = loss_class * src_logits.shape[1]
            else:
                loss_class = (
                    sigmoid_focal_loss(
                        src_logits,
                        target_classes_onehot,
                        num_boxes=num_boxes,
                        alpha=self.alpha,
                        gamma=self.gamma,
                    )
                    * src_logits.shape[1]
                )
        losses = {"loss_class": loss_class}

        return losses

    def forward(self, outputs, targets):
        outputs_without_aux = {
            k: v for k, v in outputs.items() if k != "aux_outputs" and k != "enc_outputs"
        }

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            if loss == "class":
                kwargs["GIoU_aware_class_loss"] = True if (self.training and self.GIoU_aware_class_loss) else False
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    if loss == "class":
                        kwargs["GIoU_aware_class_loss"] = True if (self.training and self.GIoU_aware_class_loss) else False
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # Compute losses for two-stage deformable-detr
        if "enc_outputs" in outputs:
            enc_outputs = outputs["enc_outputs"]
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt["labels"] = torch.zeros_like(bt["labels"])
            indices = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                kwargs = {}
                if loss == "class":
                    kwargs["GIoU_aware_class_loss"] = False
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes, **kwargs)
                l_dict = {k + "_enc": v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses
