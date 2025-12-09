# PDE


```{python}
torchrun --nproc_per_node=8 train.py --config configs/llama_3b.yaml
OMP_NUM_THREADS=6 torchrun --nproc_per_node=8 train.py --config configs/llama_1b.yaml
```