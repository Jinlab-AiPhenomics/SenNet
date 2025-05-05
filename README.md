## Introduction

SenNet is an open source semantic segmentation model based on PyTorch.

The [main](https://github.com/open-mmlab/mmsegmentation/tree/main) branch works with PyTorch 1.6+.

## Installation

Please refer to [get_started.md](docs/en/get_started.md#installation) for installation and [dataset_prepare.md](docs/en/user_guides/2_dataset_prepare.md#prepare-datasets) for dataset preparation.

## Get Started-SenNet

Please see [Overview](docs/en/overview.md) for the general introduction of MMSegmentation.

Please see [user guides](https://mmsegmentation.readthedocs.io/en/latest/user_guides/index.html#) for the basic usage of MMSegmentation.
There are also [advanced tutorials](https://mmsegmentation.readthedocs.io/en/latest/advanced_guides/index.html) for in-depth understanding of mmseg design and implementation .

Run the tools/train.py file and fill in the configuration file to train the model.

We also provide code for fitting senescence curves and calculating dynamic senescence traits and in tools/senescence-logist.py

Moreover, We provide code for machine learning to screen high-yielding varieties in tools/RF.py

## Citation

Our research is cited as fllow:

```bibtex
@misc{mmseg2020,
    title={{MMSegmentation}: OpenMMLab Semantic Segmentation Toolbox and Benchmark},
    author={MMSegmentation Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmsegmentation}},
    year={2020}
}
```

## License

This project is released under the [Apache 2.0 license](LICENSE).
