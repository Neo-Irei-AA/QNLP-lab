# QNLP-lab
This repository contains code, datasets, and experimental results produced in the course of research on Quantum Natural Language Processing (QNLP).

## Environment Set

The recommended way to reproduce the environment is to use a Python virtual environment and install dependencies via `requirements.txt`.

### 1. Clone the repository
```bash
git clone https://github.com/Neo-Irei-AA/QNLP-lab.git
cd QNLP-lab
```

### 2. Create a virtual environment
```bash
conda create -n lambeq-env python=3.10
```

Activate it:
```bash
conda activate lambeq-env
```

(Optional) Upgrade pip:
```bash
pip install --upgrade pip
```
### 3. Install dependencies
```bash
pip install -r requirements.txt
```
### 4. Verify installation
```bash
python -c "import lambeq; print(lambeq.__version__)"
```

If no error occurs, the environment setup is complete.

### Alternative: Using Conda

If you prefer Conda, you can create a Conda environment and install dependencies via pip:
```bash
conda create -n lambeq-env python=3.10
conda activate lambeq-env
pip install -r requirements.txt
```

## Notes

The experiments were developed mainly on Ubuntu.

Python 3.10 is recommended.

Some libraries (e.g. PyTorch) may require additional configuration depending on CPU/GPU availability.

For CPU-only environments, you may need:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```
## Contents

Jupyter notebooks for QNLP experiments

Reader / parser implementations

Experimental data and results

Supporting scripts and diagrams

## License

This repository is intended for academic and research purposes.
Please contact the author if you plan to reuse or extend the code.
