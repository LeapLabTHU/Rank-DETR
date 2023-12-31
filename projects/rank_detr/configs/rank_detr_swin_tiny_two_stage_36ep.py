from .rank_detr_r50_50ep import train, dataloader, optimizer, model
from detrex.config import get_config
from detectron2.config import LazyCall as L
from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone import SwinTransformer

lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_36ep
# modify model config
model.backbone = L(SwinTransformer)(
    pretrain_img_size=224,
    embed_dim=96,
    depths=(2, 2, 6, 2),
    num_heads=(3, 6, 12, 24),
    drop_path_rate=0.2,
    window_size=7,
    out_indices=(1, 2, 3),
)

# modify neck config
model.neck.input_shapes = {
    "p1": ShapeSpec(channels=192),
    "p2": ShapeSpec(channels=384),
    "p3": ShapeSpec(channels=768),
}
model.neck.in_features = ["p1", "p2", "p3"]
model.with_box_refine = True
model.as_two_stage = True

model.transformer.encoder.use_checkpoint = True
model.transformer.decoder.use_checkpoint = True

model.rank_adaptive_classhead = True
model.transformer.decoder.query_rank_layer = True
model.criterion.GIoU_aware_class_loss = True
model.criterion.matcher.iou_order_alpha = 4.0
model.criterion.matcher.matcher_change_iter = 202500

# modify training config
train.init_checkpoint = "/mnt/pretrained_backbone/swin_tiny_patch4_window7_224.pth"
train.output_dir = "./output/rank_detr_swin_tiny_two_stage_36ep"
train.max_iter = 270000
