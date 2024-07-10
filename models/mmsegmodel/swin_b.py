backbone_norm_cfg = dict(type='LN', requires_grad=True)
checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_base_patch4_window7_224_22k_20220317-4f79f7c0.pth'  # noqa

backbone=dict(
    type='SwinTransformer',
    init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
    pretrain_img_size=384,
    embed_dims=128,
    patch_size=4,
    window_size=7,
    mlp_ratio=4,
    depths=[2, 2, 18, 2],
    num_heads=[4, 8, 16, 32],
    strides=(4, 2, 2, 2),
    out_indices=(3,),
    qkv_bias=True,
    qk_scale=None,
    patch_norm=True,
    drop_rate=0.,
    attn_drop_rate=0.,
    drop_path_rate=0.3,
    use_abs_pos_embed=False,
    act_cfg=dict(type='GELU'),
    norm_cfg=backbone_norm_cfg)
