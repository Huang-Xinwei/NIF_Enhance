import os
import torch
import argparse
from utils.guidedfilter import guided_filter
from PIL import Image
from model_arch.ColorNet import ColorNet
from model_arch.ColorCGNet import ColorCGNet
from model_arch.ColorFSNet import ColorFSNet
from model_arch.ColorPSPNet import ColorPSPNet
import numpy as np
import cv2

def get_args():
    parser = argparse.ArgumentParser(description='correct input wide-spectral images color.')
    parser.add_argument('--model_dir', '-m', default='./models/', dest='model_dir', help="Specify the directory of the trained model.")
    parser.add_argument('--model_name', '-n', default='Unet', dest='molname', help="inference specific model")
    parser.add_argument('--input_dir', '-i', default='./validation/input_test/', dest='input_dir',help='Input image directory')
    parser.add_argument('--output_dir', '-o', default='./result_images/', dest='out_dir', help='Directory to save the output images')
    parser.add_argument('--device', '-d', default='cuda', dest='device', help="Device: cuda or cpu.")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    model_dir =  args.model_dir + args.molname + '.pth'
    if not os.path.exists(args.out_dir + args.molname):
        os.mkdir(args.out_dir + args.molname)
    output_img_dir = args.out_dir + args.molname
    device = torch.device(args.device)
    net = ColorNet()
    # net = ColorCGNet()
    # net = ColorFSNet()
    # net = ColorPSPNet()

    net.to(device=device)
    net.load_state_dict(torch.load(model_dir))
    net.eval()


    for file in os.listdir(args.input_dir):
        img_dir = os.path.join(args.input_dir, file)
        img = Image.open(img_dir)
        guided_image = cv2.imread(img_dir, -1)
        guided_image = guided_image[:,:576,:]/255
        img = img.crop((0, 0, 576, 320))
        image_resized = np.array(img)
        img = image_resized.transpose((2, 0, 1))
        img = img / 255
        img = torch.from_numpy(img)
        img = img.unsqueeze(0)
        img = img.to(device=device, dtype=torch.float32)

        with torch.no_grad():
            output = net(img)
        output = np.squeeze(output.cpu().numpy())
        output = output.transpose((1, 2, 0))
        output = output[:, :, ::-1]
        output_img = np.clip(output, 0, 1)
        output_b = guided_filter(guided_image, output_img[:,:,0])
        output_g = guided_filter(guided_image, output_img[:,:,1])
        output_r = guided_filter(guided_image, output_img[:,:,2])
        output_img = np.stack((output_b, output_g, output_r), axis=2) * 255

        _, fname = os.path.split(img_dir)
        name, _ = os.path.splitext(fname)

        cv2.imwrite(os.path.join(output_img_dir, name + '_' + args.molname + '.png'), output_img)



