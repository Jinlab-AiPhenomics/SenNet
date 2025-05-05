# Tutorial 2: Prepare datasets

Our dataset is organized in the form of pascal_VOC. The specific file structure is as follows:



```none
mmsegmentation
├── data
│   ├── WheatSen2023
│   │   ├── JPEGImages
│   │   ├── SegmentationClass
│   │   ├── SegmentationClassAug
│   │   ├── ImageSets
│   │   │   ├── Segmentation
|   |   |   |  ├──aug.txt
|   |   |   |  ├──train.txt
|   |   |   |  ├──val.txt
|   |   |   |  ├──tset.txt
|   |   |   |  ├──trainval.txt
```

## Pascal VOC

Pascal VOC 2012 could be downloaded from [here](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar).
Beside, most recent works on Pascal VOC dataset usually exploit extra augmentation data, which could be found [here](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz).

If you would like to use augmented VOC dataset, please run following command to convert augmentation annotations into proper format.

```shell
# --nproc means 8 process for conversion, which could be omitted as well.
python tools/dataset_converters/voc_aug.py data/VOCdevkit data/VOCdevkit/VOCaug --nproc 8
```

Please refer to [concat dataset](../advanced_guides/add_datasets.md#concatenate-dataset) and [voc_aug config example](../../../configs/_base_/datasets/pascal_voc12_aug.py) for details about how to concatenate them and train them together.

## WheatSen2023

For data acquisition, please contact jschaon@njau.edu.cn
