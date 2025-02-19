# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
from typing import Type

import mmcv
import torch
import torch.nn as nn
import os
from mmengine.model import revert_sync_batchnorm
from mmengine.structures import PixelData
from tqdm import tqdm

from mmseg.apis import inference_model, init_model
from mmseg.structures import SegDataSample
from mmseg.utils import register_all_modules
from mmseg.visualization import SegLocalVisualizer
from PIL import Image


class Recorder:
    """record the forward output feature map and save to data_buffer."""

    def __init__(self) -> None:
        self.data_buffer = list()

    def __enter__(self, ):
        self._data_buffer = list()

    def record_data_hook(self, model: nn.Module, input: Type, output: Type):
        self.data_buffer.append(output)

    def __exit__(self, *args, **kwargs):
        pass


def visualize(args, model, recorder, result, image):
    seg_visualizer = SegLocalVisualizer(
        vis_backends=[dict(type='WandbVisBackend')],
        save_dir='temp_dir',
        alpha=0.5)
    seg_visualizer.dataset_meta = dict(
        classes=model.dataset_meta['classes'],
        palette=model.dataset_meta['palette'])

    name = image.split('/')[1]
    image = mmcv.imread(image, 'color')

    seg_visualizer.add_datasample(
        name='predict',
        image=image,
        data_sample=result,
        draw_gt=False,
        draw_pred=True,
        wait_time=0,
        out_file=None,
        show=False)

    # add feature map to wandb visualizer
    for i in range(len(recorder.data_buffer)):
        feature = recorder.data_buffer[i][0]  # remove the batch
        drawn_img = seg_visualizer.draw_featmap(
            feature, image, channel_reduction='select_max')
        drawn_img = Image.fromarray(drawn_img)
        if not os.path.exists(args.out_file):
            # 使用 os.makedirs() 创建文件夹，如果路径不存在则会递归创建
            os.makedirs(args.out_file)
        # drawn_img.save(os.path.join(args.out_file, name+str(i)+'.jpg'))
        drawn_img.save(os.path.join(args.out_file, name))
        seg_visualizer.add_image(f'feature_map{i}', drawn_img)

    if args.gt_mask:
        sem_seg = mmcv.imread(args.gt_mask, 'unchanged')
        sem_seg = torch.from_numpy(sem_seg)
        gt_mask = dict(data=sem_seg)
        gt_mask = PixelData(**gt_mask)
        data_sample = SegDataSample()
        data_sample.gt_sem_seg = gt_mask

        seg_visualizer.add_datasample(
            name='gt_mask',
            image=image,
            data_sample=data_sample,
            draw_gt=True,
            draw_pred=False,
            wait_time=0,
            out_file=None,
            show=False)

    seg_visualizer.add_image('image', image)


def main():
    parser = ArgumentParser(
        description='Draw the Feature Map During Inference')
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--gt_mask', default=None, help='Path of gt mask file')
    parser.add_argument('--out-file', default=None, help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.7,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument(
        '--title', default='result', help='The image identifier.')
    args = parser.parse_args()

    register_all_modules()

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    if args.device == 'cpu':
        model = revert_sync_batchnorm(model)

    # show all named module in the model and use it in source list below
    for name, module in model.named_modules():
        print(name)

    source = [
        # 'decode_head.0.scale_heads.3.5'
        # 'backbone.layers.11.ffn.gamma2'
        # 'decode_head.bottleneck.activate',3
        # 'decode_head.psp_modules.3',
        'decode_head.bottleneck.activate',
        # 'decode_head.fpn_bottleneck.activate'
        # 'decode_head.fusion_conv.activate',
        # 'decode_head.aspp_conv.activate'
        # 'decode_head.conv_cat.activate'
        # 'decode_head.0.scale_heads.3.4.activate',
        # 'decode_head.1.bottleneck.activate',
    ]
    # source = [
    #     'decode_head.scale_heads.3.4.activate'
    # ]

    source = dict.fromkeys(source)

    count = 0
    recorder = Recorder()
    # registry the forward hook
    for name, module in model.named_modules():
        if name in source:
            count += 1
            module.register_forward_hook(recorder.record_data_hook)
            if count == len(source):
                break
    image = os.listdir(args.img)
    for img in tqdm(image):
        img = os.path.join(args.img, img)
        with recorder:
            # test a single image, and record feature map to data_buffer
            result = inference_model(model, img)

        visualize(args, model, recorder, result, img)


if __name__ == '__main__':
    main()
