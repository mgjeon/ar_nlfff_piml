- [PyTorch](https://pytorch.org/)
- [Neural Operator](https://github.com/neuraloperator/neuraloperator)

```
pip install -r requirements.txt
pip install -e .
```

```
python -m rtmag.main_dataset --config configs/dataset/<config_name>.json
python -m rtmag.main_train --config configs/train/<config_name>.json
python -m rtmag.main_test --config configs/test/<config_name>.json
```