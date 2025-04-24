#!/usr/bin/env python
# train.py

from monai.apps.auto3dseg import AutoRunner

def main():
    runner = AutoRunner(
        input="task.yaml",
        work_dir="workdir",
        algos=["segresnet"],
    )
    runner.mlflow_tracking_uri = ""
    runner.mlflow_experiment_name = ""
    runner.set_num_fold(3)

    runner.set_analyze_params({"device": "cpu", "do_ccp": False})

    runner.set_device_info(cuda_visible_devices="0")

    #  - num_workers=0 disables multiprocessing (Windows pickle fix)
    runner.set_training_params({
    "auto_scale_allowed": False,
    "num_iterations": 500,
    "num_iterations_per_validation": 50,
    "num_epochs": 500,
    "num_epochs_per_validation": 50,
    "amp": True,
    "num_workers": 0,
})

    runner.run()

if __name__ == "__main__":
    main()