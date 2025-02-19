# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
from tqdm import tqdm
from mmengine.model import revert_sync_batchnorm

from mmseg.apis import inference_model, init_model, show_result_pyplot
import os
from PIL import Image

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
    breed_list = os.listdir(args.img)
    # exist_list = os.listdir("predict/corn-orderbybreed")

    for breed in breed_list:
        # if breed in exist_list:
        #     continue
        img_list = os.listdir(os.path.join(args.img, breed))
        out_file = os.path.join(args.out_file, breed)
        if not os.path.exists(out_file):
            os.makedirs(out_file)
        i = 1
        for img in tqdm(img_list):
            # img = str(i) + ".jpg"
            out_name = os.path.join(out_file, str(i)+".jpg")
            img = os.path.join(args.img, breed, img)
            result = inference_model(model, img)
            # show the results
            res_img = show_result_pyplot(
                model,
                img,
                result,
                title=args.title,
                opacity=args.opacity,
                draw_gt=False,
                show=False if args.out_file is not None else True,
                out_file=out_name[:-3]+"png")
            res = Image.fromarray(res_img)
            res.save(out_name[:-4] + "_1.png")
            i += 1
        print("处理完" + str(breed) + "品种")



if __name__ == '__main__':
    main()
