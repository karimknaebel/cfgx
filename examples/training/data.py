config = {
    "data": {
        "datasets": {
            "hypersim": {
                "path": "data/hypersim",
                "label_type": "synthetic",
                "weight": 1,
                "image_augmentation": ["flipping", "jittering", "blurring"],
            },
            "structured3d": {
                "path": "data/structured3d",
                "label_type": "synthetic",
                "weight": 2,
            },
            "blended_mvs": {
                "path": "data/blended_mvs",
                "label_type": "sfm",
                "weight": 1,
            },
        },
    },
}
