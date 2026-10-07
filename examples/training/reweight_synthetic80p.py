from cfgx import computed, value


def reweight(get):
    datasets = {
        name: {
            "weight": get(value[name].weight),
            "label_type": get(value[name].label_type),
        }
        for name in get(value.keys())
    }
    total_weight = sum(dataset["weight"] for dataset in datasets.values())
    synthetic_weight = sum(
        dataset["weight"]
        for dataset in datasets.values()
        if dataset["label_type"] == "synthetic"
    )
    nonsynthetic_weight = total_weight - synthetic_weight
    if synthetic_weight == 0 or nonsynthetic_weight == 0:
        return {}

    return {
        name: {
            "weight": dataset["weight"]
            * (
                0.8 * total_weight / synthetic_weight
                if dataset["label_type"] == "synthetic"
                else 0.2 * total_weight / nonsynthetic_weight
            )
        }
        for name, dataset in datasets.items()
    }


config = {"data": {"datasets": computed(reweight)}}
