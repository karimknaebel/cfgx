from cfgx import final

config = {
    "model": {
        "type": "DepthEstimator",
        "backbone": {"type": "vit_small", "width": 384},
        "decoder": {"in_channels": final.model.backbone.width},
    },
}
