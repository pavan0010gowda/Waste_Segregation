# RTX 3050 training

Hardware inspected: NVIDIA GeForce RTX 3050 Laptop GPU, 4,096 MiB VRAM.
Windows driver 581.57; Ubuntu WSL can enumerate the GPU. The current Windows
Python 3.10 / TensorFlow 2.21 installation exposes only a CPU. Ubuntu currently
has Python 3.14; use a supported Python such as 3.12 in an isolated environment
after checking TensorFlow's wheel support.

Follow the [official TensorFlow WSL2 guide](https://www.tensorflow.org/install/pip).
Do not install a Linux display driver inside WSL. With a supported Python available:

```bash
cd /mnt/c/Users/pavan/waste-classification-per
python3.12 -m venv ~/.venvs/wastewise
source ~/.venvs/wastewise/bin/activate
python -m pip install --upgrade pip
python -m pip install 'tensorflow[and-cuda]>=2.16,<2.22' -r requirements.txt
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
python -m wastewise.finetune --manifest artifacts/augmented-manifest.json --batch-size 8 --epochs 12
```

GPU use must be verified by a nonempty TensorFlow GPU device list. Reduce batch
size to 4 if memory is exhausted. Close GPU-intensive applications. Fine-tuning
writes to `artifacts/runs/finetuned`; it does not replace the default deployed model.
Compare candidates using validation data, freeze the selection, then use a new
unseen test set for final claims. The optional fine-tuning workflow is separate
from the executed frozen-feature experiment; do not claim GPU speedups unless measured.
