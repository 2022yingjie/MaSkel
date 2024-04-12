import os
from PIL import Image
import numpy as np
import argparse
import albumentations

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

# from furnace import MaskingGenerator, RandomMaskingGenerator
from timm.models.vision_transformer import PatchEmbed, Block

from codebook import Codebook
from module import Encoder, Decoder

class DataAugmentationMySelf(Dataset):
    def __init__(self, args,size=0):
        self.images = [os.path.join(args.dataset_path, file) for file in os.listdir(args.dataset_path)]
        self._length = len(self.images)
        self.size = size
        # self.rescaler = albumentations.SmallestMaxSize(max_size=self.size)
        # self.cropper = albumentations.CenterCrop(height=self.size, width=self.size)
        # self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])

        random_scale = round(np.random.uniform(low=0.4, high=1.0), 2)

        self.cropper = albumentations.RandomCrop(height=int(self.size * random_scale), width=int(self.size * random_scale))
        self.rescaler = albumentations.SmallestMaxSize(max_size=self.size)
        
        self.preprocessor = albumentations.Compose([self.cropper, self.rescaler])

    def __len__(self):
        return self._length
    
    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]

        # 白底1，黑前景0
        image = (image / 255).astype(np.float32)
        
        image = image.transpose(2, 0, 1)
        return image

    def __getitem__(self, i):
        example = self.preprocess_image(self.images[i])
        return example


def load_data(args):
    train_data =  DataAugmentationMySelf(args, size=256)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False)
    return train_loader



class VQMAE(nn.Module):
    def __init__(self, args, img_size=256, patch_size=16, in_chans=3,
                 embed_dim=512, depth=8, num_heads=8,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, mask_ratio=0.0) -> None:
        super(VQMAE, self).__init__()
        # MAE encoder specifics
        self.patch_size = patch_size

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim).to(device=args.device)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim), requires_grad=False).to(device=args.device)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim)).to(device=args.device)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim), requires_grad=False).to(device=args.device)  # fixed sin-cos embedding

        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2, bias=True).to(device=args.device) # decoder to patch  * in_chans

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer).to(device=args.device)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim).to(device=args.device)


        self.decoder = Decoder().to(device=args.device)
        self.codebook = Codebook(args).to(device=args.device)
        self.quant_conv = nn.Conv2d(args.latent_dim, args.latent_dim, 1).to(device=args.device)
        self.post_quant_conv = nn.Conv2d(args.latent_dim, args.latent_dim, 1).to(device=args.device)
        self.mask_ratio = mask_ratio
    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore
    def encode_forward(self, x):
        patch_x = self.patch_embed(x) # [b, num_patches, patch_emb]
        posed_x = patch_x  + self.pos_embed
        # print(posed_x.shape)

        x_masked, mask, ids_restore = self.random_masking(posed_x, mask_ratio=self.mask_ratio)# 0.25

        for blk in self.blocks:
            x = blk(x_masked)
        x = self.norm(x)
        
        return x_masked, mask, ids_restore
    def encode_reshape_forward(self, x, ids_restore):
        # append mask tokens to sequence
        b, _, _ = x.shape
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1) 
        x_ = torch.cat([x[:, :, :], mask_tokens], dim=1) # (1,257,512)                                                                                         
        # print("shape of x_:{}".format(x_.shape))
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])) 
        # print(x_.shape) # torch.Size([1, 256, 512])                                                                                                                                                   

        # add pos embed
        x = x_ + self.decoder_pos_embed

        x = self.decoder_pred(x)
        x = x.reshape(b,-1,self.patch_size, self.patch_size)
        return x

    def forward(self, x, temb=None):
        x_masked, mask, ids_restore = self.encode_forward(x)
        quant_conv_encoded_images  = self.encode_reshape_forward(x_masked, ids_restore)
        codebook_mapping, codebook_indices, q_loss = self.codebook(quant_conv_encoded_images)
        post_quant_conv_mapping = self.post_quant_conv(codebook_mapping)
        decoded_images = self.decoder(post_quant_conv_mapping)

        return decoded_images, codebook_indices, q_loss,quant_conv_encoded_images
    @staticmethod
    def adopt_weight(disc_factor, i, threshold, value=0.):
        if i < threshold:
            disc_factor = value
        return disc_factor






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VQGAN")
    
    

    parser = argparse.ArgumentParser(description="VQGAN")
    parser.add_argument('--latent-dim', type=int, default=256, help='Latent dimension n_z (default: 256)')
    parser.add_argument('--image-size', type=int, default=256, help='Image height and width (default: 256)')
    parser.add_argument('--num-codebook-vectors', type=int, default=1024, help='Number of codebook vectors (default: 256)')
    parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss scalar (default: 0.25)')
    parser.add_argument('--image-channels', type=int, default=3, help='Number of channels of images (default: 3)')
    parser.add_argument('--dataset-path', type=str, default='/data', help='Path to data (default: /data)')
    parser.add_argument('--device', type=str, default="cpu", help='Which device the training is on')
    parser.add_argument('--batch-size', type=int, default=1, help='Input batch size for training (default: 6)')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train (default: 50)')
    parser.add_argument('--learning-rate', type=float, default=2.25e-05, help='Learning rate (default: 0.0002)')
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam beta param (default: 0.0)')
    parser.add_argument('--beta2', type=float, default=0.9, help='Adam beta param (default: 0.999)')
    parser.add_argument('--disc-start', type=int, default=10000, help='When to start the discriminator (default: 0)')
    parser.add_argument('--disc-factor', type=float, default=1., help='')
    parser.add_argument('--rec-loss-factor', type=float, default=1., help='Weighting factor for reconstruction loss.')
    parser.add_argument('--perceptual-loss-factor', type=float, default=1., help='Weighting factor for perceptual loss.')
    parser.add_argument('--patch_size', type=int, default=16)
    
    args = parser.parse_args()
    args.dataset_path = r"D:/Code/XRAY/ResBlock/rec_eval"

    loader = load_data(args)
    model = MAE(args)
    for i, data in enumerate(loader):
        model(data)


        assert 1==2
        
