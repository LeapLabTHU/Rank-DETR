from .rank_detr_r50_50ep import train, dataloader, optimizer, model
from detrex.config import get_config

lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_12ep
# modify model config
model.with_box_refine = True
model.as_two_stage = True

model.rank_adaptive_classhead = True
model.transformer.decoder.query_rank_layer = True
model.criterion.GIoU_aware_class_loss = True
model.criterion.matcher.iou_order_alpha = 4.0
model.criterion.matcher.matcher_change_iter = 67500

# modify training config
train.init_checkpoint = "detectron2://ImageNetPretrained/torchvision/R-50.pkl"
train.output_dir = "./output/rank_detr_r50_two_stage_12ep"
train.max_iter = 90000
