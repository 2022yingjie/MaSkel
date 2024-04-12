import os
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils
# from transformer import VQGANTransformer
from utils import load_data, plot_images

from MAE import MAE
from module import Encoder

class MAEEncoder(nn.Module):
    def __init__(self, args):
        super(MAEEncoder, self).__init__()
        self.model = MAE(args, mask_ratio=0.25).to(device=args.device)

        self.optim = self.configure_optimizers()
         
        self.train(args)

    def configure_optimizers(self):
        lr = args.learning_rate
        opt_vq = torch.optim.Adam(
           list(self.model.parameters()),
            lr=lr, eps=1e-08, betas=(args.beta1, args.beta2)
        )

        return opt_vq


    
    def train(self, args):
        train_dataset = load_data(args,mode='train')
        val_dataset = load_data(args, mode='val')

        for epoch in range(args.epochs):
            with tqdm(range(len(train_dataset))) as pbar:
                for i, data in zip(pbar, train_dataset):
                    imgs, _ = data
                    self.optim.zero_grad()
                    imgs = imgs.to(device=args.device)
                    
                    # latent feature
                    x_rec, x_rec_emb = self.model(imgs)
                    print("shape of x_rec:", x_rec.shape)
                    print("shape of x_rec_emb:", x_rec_emb.shape)
                    

                    loss = (imgs - x_rec)**2
                    loss = loss.mean()
                    loss.backward()
                    self.optim.step()
                    pbar.set_postfix(Light_Loss=np.round(loss.cpu().detach().numpy().item(), 4))
                    pbar.update(0)
            
                with torch.no_grad():
                    VAL_LOSS = 0.
                    for data in  val_dataset:
                        val_data, _ = data
                        val_data = val_data.to(device=args.device)

                        val_x_rec = self.model(val_data)
                        
                        loss = (val_data - val_x_rec)**2
                        loss = loss.mean()
                        VAL_LOSS += loss.item()
                    curr_val_loss = VAL_LOSS / len(val_dataset)
                    print("Saving model...")
                    print("Current Val loss: {}".format(curr_val_loss))
                    torch.save(self.model.state_dict(), os.path.join("checkpoints", f"mae_encoder_{epoch}.pt"))
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VQGAN")
    parser.add_argument('--latent-dim', type=int, default=256, help='Latent dimension n_z.')
    parser.add_argument('--image-size', type=int, default=256, help='Image height and width.)')
    parser.add_argument('--num-codebook-vectors', type=int, default=1024, help='Number of codebook vectors.')
    parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss scalar.')
    parser.add_argument('--image-channels', type=int, default=3, help='Number of channels of images.')
    parser.add_argument('--dataset-path', type=str, default='./data', help='Path to data.')
    parser.add_argument('--checkpoint-path', type=str, default='./checkpoints/last_ckpt.pt', help='Path to checkpoint.')
    parser.add_argument('--device', type=str, default="cuda", help='Which device the training is on')
    parser.add_argument('--batch-size', type=int, default=4, help='Input batch size for training.')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train.')
    parser.add_argument('--learning-rate', type=float, default=2.25e-05, help='Learning rate.')
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam beta param.')
    parser.add_argument('--beta2', type=float, default=0.9, help='Adam beta param.')
    parser.add_argument('--disc-start', type=int, default=10000, help='When to start the discriminator.')
    parser.add_argument('--disc-factor', type=float, default=1., help='Weighting factor for the Discriminator.')
    parser.add_argument('--l2-loss-factor', type=float, default=1., help='Weighting factor for reconstruction loss.')
    parser.add_argument('--perceptual-loss-factor', type=float, default=1., help='Weighting factor for perceptual loss.')

    parser.add_argument('--pkeep', type=float, default=0.5, help='Percentage for how much latent codes to keep.')
    parser.add_argument('--sos-token', type=int, default=0, help='Start of Sentence token.')

    args = parser.parse_args()
    args.dataset_path = r"/content/drive/MyDrive/X_ray_Image"

    train_transformer = MAEEncoder(args)
    train_transformer.train()
