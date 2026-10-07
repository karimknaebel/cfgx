from cfgx import computed, value

keep = ["flipping", "jittering"]


def filter_datasets(get):
    return {
        name: {
            "image_augmentation": [
                x for x in get(value[name].image_augmentation) if x in keep
            ]
        }
        for name in get(value.keys())
        if get(value[name].image_augmentation.exists())
    }


config = {
    "data": {
        "image_augmentation": value.default(()).map(
            lambda x: [item for item in x if item in keep]
        ),
        "datasets": computed(filter_datasets),
    }
}
