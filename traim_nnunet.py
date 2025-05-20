from nnunetv2.run.run_training import run_training
import os
run_training(
    configuration='2d',
    trainer_class_name='nnUNetTrainer',
    dataset_name_or_id=1, (1 вместо "Dataset001_MYTASK")
    fold=0,
    plans_identifier='nnUNetPlans',
)
