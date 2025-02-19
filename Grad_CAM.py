import os
from argparse import ArgumentParser
from pprint import pprint

import numpy as np
import torch
import torch.nn.functional as F
from mmengine import Config
from mmengine.model import revert_sync_batchnorm
from PIL import Image
from pytorch_grad_cam import GradCAM, LayerCAM, XGradCAM, GradCAMPlusPlus, EigenCAM, EigenGradCAM
from pytorch_grad_cam.utils.image import preprocess_image, show_cam_on_image
from tqdm import tqdm

from mmseg.apis import inference_model, init_model, show_result_pyplot
from mmseg.utils import register_all_modules


class SemanticSegmentationTarget:
    """wrap the model.

    requirement: pip install grad-cam

    Args:
        category (int): Visualization class.
        mask (ndarray): Mask of class.
        size (tuple): Image size.
    """

    def __init__(self, category, mask, size):
        self.category = category
        self.mask = torch.from_numpy(mask)
        self.size = size
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()

    def __call__(self, model_output):
        model_output = torch.unsqueeze(model_output, dim=0)
        model_output = F.interpolate(
            model_output, size=self.size, mode='bilinear')
        model_output = torch.squeeze(model_output, dim=0)

        return (model_output[self.category, :, :] * self.mask).sum()


def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--out-file',
        default='CAM/prediction.png',
        help='Path to output prediction file')
    parser.add_argument(
        '--cam-file', default='CAM/vis_cam.png', help='Path to output cam file')
    parser.add_argument(
        '--target-layers',
        default='decode_head.sep_bottleneck[1].pointwise_conv.activate',
        help='Target layers to visualize CAM')
    parser.add_argument(
        '--category-index', default='1', help='Category to visualize CAM')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    register_all_modules()
    model = init_model(args.config, args.checkpoint, device=args.device)
    if args.device == 'cpu':
        model = revert_sync_batchnorm(model)
    pprint([name for name, _ in model.named_modules()])
    # test a single image

    img_list = os.listdir(args.img)
    for img in tqdm(img_list):
        temp = img
        img = os.path.join(args.img, img)
        result = inference_model(model, img)
        # show the results
        show_result_pyplot(
            model,
            img,
            result,
            draw_gt=False,
            show=False ,
            # if args.out_file is not None else True,
            out_file=os.path.join(args.out_file))

        # result data conversion
        prediction_data = result.pred_sem_seg.data
        pre_np_data = prediction_data.cpu().numpy().squeeze(0)

        target_layers = args.target_layers
        target_layers = [eval(f'model.{target_layers}')]

        category = int(args.category_index)
        mask_float = np.float32(pre_np_data == category)

        # data processing
        image = np.array(Image.open(img).convert('RGB'))
        height, width = image.shape[0], image.shape[1]
        rgb_img = np.float32(image) / 255
        config = Config.fromfile(args.config)
        image_mean = config.data_preprocessor['mean']
        image_std = config.data_preprocessor['std']
        input_tensor = preprocess_image(
            rgb_img,
            mean=[x / 255 for x in image_mean],
            std=[x / 255 for x in image_std])

        # Grad CAM(Class Activation Maps)
        # Can also be LayerCAM, XGradCAM, GradCAMPlusPlus, EigenCAM, EigenGradCAM
        targets = [
            SemanticSegmentationTarget(category, mask_float, (height, width))
        ]
        with EigenCAM(
                model=model,
                target_layers=target_layers) as cam:  # Removed use_cuda parameter
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            savepath = os.path.join(args.cam_file, temp)
            # save cam file
            Image.fromarray(cam_image).save(savepath)


if __name__ == '__main__':
    main()
