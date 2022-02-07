import itertools
import subprocess

lrs = [0.01, 0.03]
optimizers = ["adam", "adamax"]

for lr, opt in itertools.product(lrs, optimizers):
    subprocess.run(
        [
            "dvc",
            "exp",
            "run",
            "--queue",
            "--set-param",
            f"optimizer={opt}",
            "--set-param",
            f"learning_rate={lr}",
            "train",
        ]
    )
