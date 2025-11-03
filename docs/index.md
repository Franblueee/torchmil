---
description: torchmil is a PyTorch library for building Multiple Instance Learning (MIL) models with ease. Learn how to implement MIL in PyTorch with torchmil.
---

# torchmil

**torchmil** is a [PyTorch](https://pytorch.org/)-based library for deep Multiple Instance Learning (MIL).
It provides a simple, flexible, and extensible framework for working with MIL models and data.

It includes:

- A collection of popular [MIL models](api/models/index.md).
- Different [PyTorch modules](api/nn/index.md) frequently used in MIL models.
- Handy tools to deal with [MIL data](api/data/index.md).
- A collection of popular [MIL datasets](api/datasets/index.md).

## Installation

```bash
pip install torchmil
```

## Quick start

You can load a MIL dataset and train a MIL model in just a few lines of code:

```python
from torchmil.datasets import Camelyon16MIL
from torchmil.models import ABMIL
from torchmil.utils import Trainer
from torchmil.data import collate_fn
from torch.utils.data import DataLoader

# Load the Camelyon16 dataset
dataset = Camelyon16MIL(root='data', features='UNI')
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

# Instantiate the ABMIL model and optimizer
model = ABMIL(in_shape=(2048,), criterion=torch.nn.BCEWithLogitsLoss()) # each model has its own criterion
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Instantiate the Trainer
trainer = Trainer(model, optimizer, device='cuda')

# Train the model
trainer.train(dataloader, epochs=10)

# Save the model
torch.save(model.state_dict(), 'model.pth')
```

## Next steps

You can take a look at the [examples](examples/index.md) to see how to use **torchmil** in practice.
To see the full list of available models, datasets, and modules, check the [API reference](api/index.md).

## Contributing to torchmil

We welcome contributions to **torchmil**! There several ways you can contribute:

- Reporting bugs or issues you encounter while using the library, asking questions, or requesting new features: use the [Github issues](https://github.com/Franblueee/torchmil/issues).
- Improving the documentation: if you find any part of the documentation unclear or incomplete, feel free to submit a pull request with improvements.
- If you have a new model, dataset, or utility that you think would be useful for the community, please consider submitting a pull request to add it to the library.

Take a look at [CONTRIBUTING.md](https://github.com/Franblueee/torchmil/blob/main/CONTRIBUTING.md) for more details on how to contribute.

## Citation

If you find this library useful, please consider citing it:

```bibtex
@article{castro2025torchmil,
  title={torchmil: A PyTorch-based library for deep Multiple Instance Learning},
  author={Castro-Mac{\'\i}as, Francisco M and S{\'a}ez-Maldonado, Francisco J and Morales-{\'A}lvarez, Pablo and Molina, Rafael},
  journal={arXiv preprint arXiv:2509.08129},
  year={2025}
}
```
