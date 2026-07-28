# Gemma3-12B PiSSA Fine-Tuning

This repository contains the dataset, configuration files, and training methodology used to fine-tune a Gemma-based Large Language Model for generating valid CadQuery Python scripts from natural language instructions. 

The training leverages **Principal Singular Values and Singular Vectors Adaptation (PiSSA)** for highly memory-efficient and mathematically stable parameter-efficient fine-tuning (PEFT), distributed across multi-GPU environments.

## Hardware Requirements
This specific training configuration was executed using `torchrun` and DeepSpeed on:
* **GPUs**: 2x NVIDIA RTX 6000 (48GB VRAM each)
* **Compute**: `bf16` Mixed Precision
* **Peak Memory Strategy**: Per-device batch size of 1 with gradient accumulation of 16 to bypass ZeRO optimizer memory spikes during SVD initialization.

## Reproducing the Training Run

### 1. Environment Setup
We utilize [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) to manage the DeepSpeed ZeRO partitioning and PiSSA initializations. 

```bash
git clone --depth 1 [https://github.com/hiyouga/LLaMA-Factory.git](https://github.com/hiyouga/LLaMA-Factory.git)
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```

### 2. Configure the Data
Symlink the training dataset directly into LLaMA-Factory's data directory and replace the registry:

```bash
ln -sf /path/to/text2cad_finetune.jsonl data/text2cad_finetune.jsonl
cp /path/to/your/dataset_info.json data/dataset_info.json
```

### 3. Launch Distributed Training
Execute the training run utilizing the provided configuration file. The `FORCE_TORCHRUN=1` flag ensures the workload is safely sharded across both GPUs via NCCL.

```bash
FORCE_TORCHRUN=1 NPROC_PER_NODE=2 llamafactory-cli train configs/train_pissa.yaml
```

### 4. Interactive Inference
To quickly test the generated PiSSA adapters against the base model without permanently merging the weights:

```bash
llamafactory-cli chat \
  --model_name_or_path google/gemma-2-9b \
  --adapter_name_or_path adapters/ \
  --template gemma \
  --finetuning_type pissa
```

## Results & Evaluation

The PiSSA finetune failed to run due to C++ backend crashes. Although it was unsuccessful, I am pivoting towards implementing DoRA, a different finetuning method specializing in code and learning syntax.
