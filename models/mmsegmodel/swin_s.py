backbone_norm_cfg = dict(type='LN', requires_grad=True)
checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_small_patch4_window7_224_20220317-7ba6d6dd.pth'  # noqa

backbone=dict(
    type='SwinTransformer',
    pretrain_img_size=224,
    init_cfg=dict(type='Pretrained',checkpoint=checkpoint_file,),
    embed_dims=96,
    depths=[2, 2, 18, 2],
    num_heads=[3, 6, 12, 24],
    strides=(4, 2, 2, 2),
    out_indices=(0, 1, 2, 3),
    patch_size=4,
    window_size=7,
    mlp_ratio=4,
    use_abs_pos_embed=False,
    drop_path_rate=0.3,
    patch_norm=True,
    qkv_bias=True,
    qk_scale=None,
    drop_rate=0.,
    attn_drop_rate=0.,
    act_cfg=dict(type='GELU'),
    norm_cfg=backbone_norm_cfg,
)
