import logging
import os
import sys
import torch
import argparse
import numpy as np
from torch import optim
from tqdm import tqdm
from model_arch.ColorNet import ColorNet
from utils.datasets_manger import Mydataset
from utils.loss_func import darkchannel_loss, cie76_part_loss
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter


def train_net(net,
              device,
              epochs=110,
              batch_size=32,
              lr=0.0001,
              val_percent=0.1,
              chkpointperiod=10,
              patchsz=256,
              validationFrequency=4,
              dir_img='/data/hxw/NewNIFdatasets',
              model_name='ColorXNet',
              save_cp=True):
    dataset = Mydataset(dir_img, patch_size=patchsz, patch_num_per_images=1)
    mse_loss = torch.nn.MSELoss(reduction='sum')
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=32, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    writer = SummaryWriter(comment=f'_{model_name}_LR_{lr}_BS_{batch_size}_EP_{epochs}')
    global_step = 0
    logging.info(f'''Starting training:
        Epochs:          {epochs} epochs
        Batch size:      {batch_size}
        Patch size:      {patchsz} x {patchsz}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Validation Frq.: {validationFrequency}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        training model:  {model_name}
    ''')
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.00001)

    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                combine_imgs = batch['input']
                rgb_imgs = batch['gt']

                combine_imgs = combine_imgs.to(device=device, dtype=torch.float32)
                rgb_imgs = rgb_imgs.to(device=device, dtype=torch.float32)

                imgs_pred = net(combine_imgs)
                loss = mse_loss(imgs_pred, rgb_imgs) + 0.125*(cie76_part_loss(imgs_pred, rgb_imgs) + darkchannel_loss(imgs_pred, rgb_imgs, window=14))
                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.update(np.ceil(combine_imgs.shape[0]))
                global_step += 1

        if (epoch + 1) % validationFrequency == 0:
            val_score = vald_net(net, val_loader, device)
            logging.info('Validation Loss: {}'.format(val_score))
            writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)
            writer.add_scalar('Loss/test', val_score, global_step)
            writer.add_images('images', combine_imgs[:,:3,:,:], global_step)
            writer.add_images('result-images', imgs_pred, global_step)
            writer.add_images('GT', rgb_imgs, global_step)

 
        if save_cp and (epoch + 1) % chkpointperiod == 0:
            if not os.path.exists('checkpoint_file'):
                os.mkdir('checkpoint_file')
                logging.info('Created checkpoint directory')

            torch.save(net.state_dict(), 'checkpoint_file/' + f'Wide_Spectral{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved!')
        if not os.path.exists('models'):
            os.mkdir('models')
            logging.info('Created trained models directory')
        torch.save(net.state_dict(), 'models/' + model_name + '.pth')
        logging.info('Saved trained models!')
        writer.close()
        logging.info('End of training')


def vald_net(net, loader, device):
    net.eval()
    n_val = len(loader) + 1
    mse_dcl = 0
    mse_loss = torch.nn.MSELoss(reduction='sum')
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            combine_imgs = batch['input']
            rgb_imgs = batch['gt']
            patchnum = combine_imgs.shape[1] / 3

            combine_imgs = combine_imgs.to(device=device, dtype=torch.float32)
            rgb_imgs = rgb_imgs.to(device=device, dtype=torch.float32)

            with torch.no_grad():
                imgs_pred = net(combine_imgs)
                loss = mse_loss(imgs_pred, rgb_imgs) + 0.125*(cie76_part_loss(imgs_pred, rgb_imgs) + darkchannel_loss(imgs_pred, rgb_imgs, window=14))
                mse_dcl = mse_dcl + loss
            pbar.update(np.ceil(combine_imgs.shape[0] / patchnum))

    net.train()
    return mse_dcl / n_val


def get_args():
    parser = argparse.ArgumentParser(description='Train Wide Spectral color correct.')
    parser.add_argument('-ep', '--epochs', dest='epochs', default=300, type=int, help='Number of epochs')
    parser.add_argument('-bs', '--batch-size', dest='batchsize', default=32, type=int, nargs='?', help='Batch size')
    parser.add_argument('-lr', '--learning-rate', dest='lr', default=0.0003, type=float, nargs='?', help='Learning rate')
    parser.add_argument('-ps', '--patch-size', dest='patchsz', default=256, type=int, help='Size of training patch')
    parser.add_argument('-vr', '--validation', dest='val', default=0.1, type=float, help='Percent of the data that is used as validation (0-1)')
    parser.add_argument('-vf', '--validation-frequency', dest='val_frq', default=5, type=int, help='Validation frequency.')
    parser.add_argument('-cp', '--checkpoint-period', dest='chkpointperiod', default=10, type=int, help='Number of epochs to save a checkpoint')
    parser.add_argument('-trd', '--training_dir', dest='trdir', default='/data/hxw/NewNIFdatasets', help='Training image directory')
    parser.add_argument('-n', '--model_name', dest='molname', default='UCGNet2.0', help='Network model name')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info('Training of wide-spectral color correction')
    args = get_args()
    device = torch.device('cuda')
    logging.info(f'Using device {device}')
    
    net = ColorNet()
    net.to(device=device)

    train_net(net=net,
                device=device,
                epochs=args.epochs,
                batch_size=args.batchsize,
                lr=args.lr,
                val_percent=args.val,
                lrdf=args.lrdf,
                lrdp=args.lrdp,
                chkpointperiod=args.chkpointperiod,
                patchsz=args.patchsz,
                validationFrequency=args.val_frq,
                dir_img=args.trdir,
                model_name=args.molname
                )



