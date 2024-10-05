<img align="right" src="https://github.com/IonutMotoi/CutPasteSatSeg/assets/32934655/9c1f2776-7d42-4a2d-8444-ac52740d5445" width=30% height=30%>

# Evaluating the Efficacy of Cut-and-Paste Data Augmentation in Semantic Segmentation for Satellite Imagery

Repository for the paper ["Evaluating the Efficacy of Cut-and-Paste Data Augmentation in Semantic Segmentation for Satellite Imagery"](https://arxiv.org/abs/2404.05693) - IEEE IGARSS 2024

## Install

The provided example makes use of PyTorch, Rasterio and the [DynamicEarthNet](https://arxiv.org/abs/2203.12560) dataset, but can be easily extended to other libraries or datasets.

```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Instance Extraction
Before using the Cut-and-Paste augmentation, you need to extract instances from your dataset. Use the `generate_cap_dataset.py` script for this purpose.
<br clear="right"/>
### Step 2: Instance Pasting

#### Extend your existing dataset class to incorporate the Cut-and-Paste augmentation:

```
from cut_and_paste import CutAndPaste

class YourDataset_CAP(YourDataset):
    """
    Your dataset class with Cut-and-Paste augmentation
    """

    def __init__(self, cfg, split):
        super().__init__(cfg, split)
        self.cut_and_paste = CutAndPaste(cfg["cut_and_paste"])

    def __getitem__(self, index):
        # Load image and label as in your original dataset class
        # They should be numpy arrays before applying the Cut-and-Paste augmentation

        # Apply Cut-and-Paste augmentation
        self.cut_and_paste.paste_instances(img, mask)

        # Rest of your code
        # ...
        return img, mask
```

#### Use the extended dataset class in your training pipeline:

```
# In your training script
dataset = YourDataset_CAP(config, split='train')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```
#### Configuration
The CutAndPaste class accepts a configuration dictionary with the following keys:

* root: Path to the extracted instances (output_folder from Step 1)
* classes: List of class indices from which to sample
* num_of_instances: Number of instances to paste per image
* augment_instances: Whether to apply pre-pasting augmentations

Example configuration:
```
cut_and_paste_config = {
    "root": "dataset/cap_dataset",
    "classes": [0, 1, 2, 3, 4, 5],
    "num_of_instances": 100,
    "augment_instances": True
}
```

## Cite
If you find this work useful in your research, please consider citing:
```
@INPROCEEDINGS{10640734,
  author={Motoi, Ionut M. and Saraceni, Leonardo and Nardi, Daniele and Ciarfuglia, Thomas A.},
  booktitle={IGARSS 2024 - 2024 IEEE International Geoscience and Remote Sensing Symposium}, 
  title={Evaluating the Efficacy of Cut-and-Paste Data Augmentation in Semantic Segmentation for Satellite Imagery}, 
  year={2024},
  volume={},
  number={},
  pages={9802-9806},
  keywords={Training;Adaptation models;Semantic segmentation;Semantics;Urban planning;Land surface;Satellite images;Semantic Segmentation;Land Use Land Cover;Deep Learning;Data Augmentation;Cut-and-Paste;Copy-Paste;Satellite Remote Sensing},
  doi={10.1109/IGARSS53475.2024.10640734}}
```
