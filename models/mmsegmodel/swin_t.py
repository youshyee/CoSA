checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_tiny_patch4_window7_224_20220317-1cdeb081.pth'  # noqa
backbone_norm_cfg = dict(type='LN', requires_grad=True)

backbone=dict(
    type='SwinTransformer',
    init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
    embed_dims=96,
    depths=[2, 2, 6, 2],
    num_heads=[3, 6, 12, 24],
    window_size=7,
    use_abs_pos_embed=False,
    drop_path_rate=0.3,
    patch_norm=True,
    pretrain_img_size=224,
    patch_size=4,
    mlp_ratio=4,
    strides=(4, 2, 2, 2),
    out_indices=(0, 1, 2, 3),
    qkv_bias=True,
    qk_scale=None,
    drop_rate=0.,
    attn_drop_rate=0.,
    act_cfg=dict(type='GELU'),
    norm_cfg=backbone_norm_cfg,
)

