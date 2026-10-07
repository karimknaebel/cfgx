import math

from cfgx import final, value

config = {
    "batch_size": 64,
    "steps": 48_000,
    "lr": value * (final.batch_size / 128).map(math.sqrt),
}
