import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import utils as vutils
from discriminator import Discriminator
from lpips import LPIPS

from MAE import MAE, VQMAE
from module import Encoder
#
from utils import load_data, weights_init


class TrainVQGAN:
    def __init__(self, args):
        self.mae = MAE(args, mask_ratio=0.0).to(device=args.device)
        self.mae.load_state_dict(torch.load("/content/drive/MyDrive/mae.pt"))
        for param in self.mae.parameters():
            param.requires_grad = False
        
        self.vqmae = VQMAE(args).to(device=args.device)
        self.discriminator = Discriminator(args).to(device=args.device)
        self.discriminator.apply(weights_init)
        self.perceptual_loss = LPIPS().eval().to(device=args.device)
        self.opt_vq, self.opt_disc = self.configure_optimizers(args)
        
        self.prepare_training()
        
        
        self.train(args)

    def configure_optimizers(self, args):
        lr = args.learning_rate
        opt_vq = torch.optim.Adam(
            self.vqmae.parameters(),
            lr=lr, eps=1e-08, betas=(args.beta1, args.beta2)
        )
        opt_disc = torch.optim.Adam(self.discriminator.parameters(),
                                    lr=lr, eps=1e-08, betas=(args.beta1, args.beta2))

        return opt_vq, opt_disc

    @staticmethod
    def prepare_training():
        os.makedirs("results", exist_ok=True)
        os.makedirs("checkpoints", exist_ok=True)
    
    def varLOSS(self, emb_vector):
        B, C,_, _  = emb_vector.shape
        emb_vector = emb_vector.reshape(B,-1)

        mean_embedding = torch.mean(emb_vector, dim=0)

        distances = torch.norm(emb_vector - mean_embedding, dim=1)
       
        loss = torch.exp(-torch.var(distances))
        return loss
    
    def train(self, args):
        train_dataset = load_data(args, mode='train')
        val_dataset = load_data(args, mode='val')
        steps_per_epoch = len(train_dataset)
        
        for epoch in range(args.epochs):
            with tqdm(range(len(train_dataset))) as pbar:
                for i, data in zip(pbar, train_dataset):
                    imgs, label = data
                    imgs = imgs.to(device=args.device)
                    label = label.to(device=args.device)
                    
                    _, xray_image_emb = self.mae(imgs)
                    
                    #### pretrained Vit Encoder
                    decoded_images, _, q_loss, mask_image_emb = self.vqmae(label)

                    emb_distance_loss = ((xray_image_emb - mask_image_emb)**2).mean()
                    
                    disc_real = self.discriminator(imgs)
                    disc_fake = self.discriminator(decoded_images)

                    disc_factor = self.vqmae.adopt_weight(args.disc_factor, epoch*steps_per_epoch+i, threshold=args.disc_start)

                    perceptual_loss = self.perceptual_loss(imgs, decoded_images)
                    # assert perceptual_loss.mean().cpu().detach().numpy().item() > 0
                    
                    # rec_loss = torch.abs(imgs - decoded_images)
                    rec_loss = (imgs - decoded_images)**2 + torch.abs(imgs - decoded_images)
                    assert rec_loss.mean().cpu().detach().numpy().item() > 0
                    
                    perceptual_rec_loss = args.perceptual_loss_factor * perceptual_loss + args.rec_loss_factor * rec_loss
                    # perceptual_rec_loss = args.rec_loss_factor * rec_loss
                    perceptual_rec_loss = perceptual_rec_loss.mean()
                    g_loss = -torch.mean(disc_fake)

                    # λ = self.vqgan.calculate_lambda(perceptual_rec_loss, g_loss)
                    vq_loss = perceptual_rec_loss + q_loss#  + disc_factor * λ * g_loss
                    
                    # vq_loss + emb_dis_loss + emb_var_loss
                    vq_loss = vq_loss + emb_distance_loss

                    d_loss_real = torch.mean(F.relu(1. - disc_real))
                    d_loss_fake = torch.mean(F.relu(1. + disc_fake))
                    gan_loss = disc_factor * 0.5*(d_loss_real + d_loss_fake)

                    self.opt_vq.zero_grad()
                    vq_loss.backward(retain_graph=True)

                    self.opt_disc.zero_grad()
                    gan_loss.backward()

                    self.opt_vq.step()
                    self.opt_disc.step()

                    if epoch % 20 == 0:
                        with torch.no_grad():
                            # real_fake_images = torch.cat((imgs[:4], decoded_images.add(1).mul(0.5)[:4]))
                            # real_fake_images = torch.cat((imgs.add(1).mul(0.5)[:4], decoded_images.add(1).mul(0.5)[:4]))
                            real_fake_images = torch.cat(((1-label)[:4] ,(1-imgs)[:4], (1-decoded_images)[:4]))
                            
                            vutils.save_image(real_fake_images, os.path.join("results", f"{epoch}_{i}.jpg"), nrow=4)

                    pbar.set_postfix(
                        VQ_Loss=np.round(vq_loss.cpu().detach().numpy().item(), 5),
                        GAN_Loss=np.round(gan_loss.cpu().detach().numpy().item()*10000, 3),
                        # Dis_Emb_Loss = np.round(emb_distance_loss.cpu().detach().numpy().item(), 5),
                        # Vit_Var_Loss = np.round(vit_emb_var_loss.cpu().detach().numpy().item(), 5)
                    )
                    pbar.update(0)
                with torch.no_grad():
                    VAL_LOSS = 0.
                    for data_synthesis in  val_dataset:
                        val_data, val_label = data_synthesis
                        val_data = val_data.to(device=args.device)
                        val_label = val_label.to(device=args.device)
                           
                        decoded_images, _, q_loss, vit_image_emb = self.vqmae(val_label)
                        
                        rec_loss = (val_data - decoded_images)**2 + torch.abs(val_data - decoded_images)
                        perceptual_loss = self.perceptual_loss(val_data, decoded_images)
                        
                        perceptual_rec_loss = args.perceptual_loss_factor * perceptual_loss + args.rec_loss_factor * rec_loss
                        loss_ = perceptual_rec_loss.mean() # + vit_emb_var_loss #emb_distance_loss#  + 
                        VAL_LOSS += loss_.item()
                    curr_val_loss = VAL_LOSS / len(val_dataset)
                    print("Saving model...")
                    print("Current Val loss: {}".format(curr_val_loss))
                    torch.save(self.vqmae.state_dict(), os.path.join("checkpoints", f"vqmae_epoch_{epoch}.pt"))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VQGAN")
    parser.add_argument('--latent-dim', type=int, default=256, help='Latent dimension n_z (default: 256)')
    parser.add_argument('--image-size', type=int, default=256, help='Image height and width (default: 256)')
    parser.add_argument('--num-codebook-vectors', type=int, default=1024, help='Number of codebook vectors (default: 256)')
    parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss scalar (default: 0.25)')
    parser.add_argument('--image-channels', type=int, default=3, help='Number of channels of images (default: 3)')
    parser.add_argument('--dataset-path', type=str, default='/data', help='Path to data (default: /data)')
    parser.add_argument('--device', type=str, default="cuda", help='Which device the training is on')
    parser.add_argument('--batch-size', type=int, default=4, help='Input batch size for training (default: 6)')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train (default: 50)')
    parser.add_argument('--learning-rate', type=float, default=2.25e-05, help='Learning rate (default: 0.0002)')# 2.25e-05
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam beta param (default: 0.0)')
    parser.add_argument('--beta2', type=float, default=0.9, help='Adam beta param (default: 0.999)')
    parser.add_argument('--disc-start', type=int, default=10000, help='When to start the discriminator (default: 0)')
    parser.add_argument('--disc-factor', type=float, default=1., help='')
    parser.add_argument('--rec-loss-factor', type=float, default=1., help='Weighting factor for reconstruction loss.')
    parser.add_argument('--perceptual-loss-factor', type=float, default=1., help='Weighting factor for perceptual loss.')
    parser.add_argument('--mode', type=str, default='train', help='')


    args = parser.parse_args()
    args.dataset_path = r"/content/drive/MyDrive/X_ray_Image"

    train_vqgan = TrainVQGAN(args)
    train_vqgan.train()


