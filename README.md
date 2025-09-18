## Introduction
Missing modalities critically undermine the reliability and generalization of multimodal learning. Existing methods either uniformly aggregate available modalities, ignoring asymmetric dependencies, or reconstruct missing ones, suffering from bias and uncertainty under distribution shifts. In this paper, we propose Pattern-Aware Routing and Prompting (PARP), a unified framework for robust incomplete-modality learning. Specifically, (1) a pattern-aware graph encoder captures missing-pattern structural dependencies for resilient representation; (2) sparse routing generates meta-adaptive prompts for instance-adaptive fusion; and (3) uncertainty-aware prompt distillation combines confidence-weighted teacher signals with student-driven graph and contrastive consistency, ensuring robust knowledge transfer and reasoning under incomplete observations. Experiments demonstrate consistent gains across diverse missing-modality settings, with notable improvements on the MM-IMDb dataset.

## Usage
### Enviroment
#### Prerequisites
Python = 3.7.13

Pytorch = 1.10.0

CUDA = 11.3

#### Other requirements
```
pip install -r requirements.txt
```

### Evaluation
```
python run.py with data_root=<ARROW_ROOT> \
        num_gpus=<NUM_GPUS> \
        num_nodes=<NUM_NODES> \
        per_gpu_batchsize=<BS_FITS_YOUR_GPU> \
        <task_finetune_mmimdb or task_finetune_food101 or task_finetune_hatememes> \
        load_path=<MODEL_PATH> \
        exp_name=<EXP_NAME> \
        prompt_type=<PROMPT_TYPE> \
        test_ratio=<TEST_RATIO> \
        test_type=<TEST_TYPE> \
        test_only=True     
```

### Train
```
python run.py with data_root=<ARROW_ROOT> \
        num_gpus=<NUM_GPUS> \
        num_nodes=<NUM_NODES> \
        per_gpu_batchsize=<BS_FITS_YOUR_GPU> \
        <task_finetune_mmimdb or task_finetune_food101 or task_finetune_hatememes> \
        load_path=<PRETRAINED_MODEL_PATH> \
        exp_name=<EXP_NAME>
```
