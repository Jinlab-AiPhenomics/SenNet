# Copyright (c) OpenMMLab. All rights reserved.
import os
from argparse import ArgumentParser

from mmengine.model import revert_sync_batchnorm

from mmseg.apis import inference_model, init_model, show_result_pyplot


def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
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

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    if args.device == 'cpu':
        model = revert_sync_batchnorm(model)
    # test a single image
    imgs = os.listdir(args.img)
    for img in imgs:
        img = os.path.join(args.img, img)
        result = inference_model(model, img)
        # show the results
        show_result_pyplot(
            model,
            img,
            result,
            title=args.title,
            opacity=args.opacity,
            draw_gt=False,
            show=False if args.out_file is not None else True,
            out_file=args.out_file)


if __name__ == '__main__':
    main()