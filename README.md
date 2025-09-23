## About This Repository

This fork is **not** a replacement for the original Basenji project.  
Its scope is limited to:

- Adding a **CLI wrapper** for Akita, where CLI wrappers can input **fasta** file, and outputs **h5** files of Akita predictions.
- Updating the environment setup to primarily use **uv**, with minimal reliance on `conda`,  
  for simpler installation and dependency management.
- Updates to the original **Basenji** codebase will not be tracked here.

---
## Citation Notice

If you use **Basenji**, **Akita**, or any other software from this repository in your work,  
please **cite the original work** (not this fork).

For further details, see the original Basenji repository:  
➡️ [https://github.com/calico/basenji](https://github.com/calico/basenji)

---
## Setup Instructions

#### Clone the repo and Sync the uv
```bash
uv sync
```

### Workaround needed to downgrade cuda to 11.8, to be compatible with older TF version.
#### Create a tiny env that only hosts the CUDA .so's TF needs
```bash
micromamba create -n cuda118 -c conda-forge \
    cudatoolkit=11.8 cudnn=8.8.0.121 -y
```

#### Then, set these env vars before running uv python:
```bash
# Replace with your actual micromamba envs path if different
export CUDA_HOME="$HOME/micromamba/envs/cuda118"
# Clear old path that was masking CUDA in your logs, then add this env's lib dir
unset LD_LIBRARY_PATH
export LD_LIBRARY_PATH="$CUDA_HOME/lib"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CUDA_HOME"
```

#### Check it worked:
```bash
uv run python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

#### Install basenji
```bash
uv pip install -e . --no-deps
```

#### For v2 models, download akita models (v2)
#### v2 model is not provided as a single checkpoint (unlike v1), akita_cli_v2 ensembles model predictions from each fold by default. Please inspect each fold's predictions individually to decide on whether to ensemble them. For more details on how the model was trained: https://github.com/calico/basenji/tree/master/manuscripts/akita/v2
```bash
uv run gsutil cp -r gs://basenji_hic/3-2021/models .
uv run akita_cli_v2.py --help
```

#### For v1 models, akita v1 model is in data.
```bash
uv run akita_cli_v1.py --help
```
