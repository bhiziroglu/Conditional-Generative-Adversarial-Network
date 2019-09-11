from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Data(Dataset):
    def __init__(self, file_name):
        batches = [self.unpickle(file_name+"data_batch_"+str(i)) for i in range(1,6)] + [self.unpickle(file_name+"test_batch")]
        self.data = np.vstack([a[b'data'] for a in batches])
        self.data = np.asarray(\
            [self.reshape_to_image(x) for x in self.data]\
        )
        self.labels = np.hstack([a[b'labels'] for a in batches])

    def unpickle(self,file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    #https://github.com/nafarya/Image-Recognition/blob/0cbb280e7c84af529ec3e44590554678897c2a3f/dataset.py
    def reshape_to_image(self, image_as_numpy_array):
        img_R = image_as_numpy_array[0:1024].reshape((32, 32))
        img_G = image_as_numpy_array[1024:2048].reshape((32, 32))
        img_B = image_as_numpy_array[2048:3072].reshape((32, 32))
        img = np.dstack((img_R, img_G, img_B))
        return img

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        image = self.data[i]
        label = self.labels[i]
        to_tensor = transforms.Compose([
            transforms.ToTensor()
        ])
        image = to_tensor(image)
        image = (image - 0.5) * 2
        return image, label

class Generator(nn.Module):
    def __init__(self, embeddings=None, nc=3, nz=100, ngf=64):
        super(Generator, self).__init__()

        self.linear = nn.Linear(200,100)
        self.embeddings = embeddings

        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False), # torch.Size([1, 512, 4, 4])
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False), # .Size([1, 256, 8, 8])
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False), # e([1, 128, 16, 16])
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False), # .Size([1, 64, 32, 32])
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, kernel_size=1, stride=1, padding=0, bias=False), # ize([1, 3, 32, 32])
            nn.Tanh()
        )

    def forward(self, x, label_embed):
        label_embed = self.embeddings(label_embed)

        x = x.view(-1,100)
        x = torch.cat([x,label_embed], dim=1)

        x = self.linear(x)
        x = x.unsqueeze(2).unsqueeze(3)

        output = self.main(x)
        return output

class Discriminator(nn.Module):
    def __init__(self, embeddings= None ,nc=3, ndf=64):
        super(Discriminator, self).__init__()
        self.embeddings = embeddings

        self.label_to_image = nn.Linear(100,32*32*3)
        self.conv1 = nn.Conv2d(nc * 2, nc, 1, 1, 0, bias=False)

        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False), # [1, 64, 16, 16])
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False), # [1, 128, 8, 8])
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False), # ([1, 256, 4, 4])
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False), # [1, 512, 2, 2])
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 2, 2, 0, bias=False), # [1,1,1,1]
            nn.Sigmoid()
        )

    def forward(self, x, label_embed):
        label_embed = self.embeddings(label_embed)

        label_map = self.label_to_image(label_embed)
        label_map = label_map.view(-1,3,32,32)

        x = torch.cat([x,label_map], dim=1)

        out = self.conv1(x)
        output = self.main(out)

        return output

def main():
    batchsize = 200
    train_data = Data(file_name="cifar-10-batches-py/")
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batchsize,shuffle=True,drop_last=True)

    embeddings = nn.Embedding(10,100)
    embeddings.weight.requires_grad = False

    netD = Discriminator(embeddings)
    netG = Generator(embeddings)
  
    netD.to(device)
    netG.to(device)

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

    netD.train()
    netG.train()

    nz = 100

    criterion = nn.BCELoss()
    criterion.to(device)
    
    real_label = torch.ones([batchsize, 1], dtype=torch.float).to(device)
    fake_label = torch.zeros([batchsize ,1], dtype=torch.float).to(device)
    
    for _ in range(500):
        index = 0
        for i, (input_sequence, label) in enumerate(train_data_loader):
            fixed_noise = torch.randn(batchsize, nz, 1, 1, device=device)

            input_sequence = input_sequence.to(device)
            label_embed = label.to(device)
            '''
                Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            '''

            D_real_result = netD(input_sequence, label_embed)
            D_real_loss = criterion(D_real_result.view(batchsize, -1), real_label)

            G_result = netG(fixed_noise, label_embed)

            D_fake_result = netD(G_result, label_embed)

            D_fake_loss = criterion(D_fake_result.view(batchsize, -1), fake_label)

            # Back propagation
            D_train_loss = (D_real_loss + D_fake_loss) / 2

            netD.zero_grad()
            D_train_loss.backward()
            optimizerD.step()

            '''
                Update G network: maximize log(D(G(z)))
            '''
            new_label = torch.LongTensor(batchsize, 10).random_(0, 10).to(device)
            new_embed = new_label[:, 0].view(-1)

            G_result = netG(fixed_noise, new_embed)

            D_fake_result = netD(G_result, new_embed)
            G_train_loss = criterion(D_fake_result.view(batchsize, -1), real_label)


            # Back propagation
            netD.zero_grad()
            netG.zero_grad()
            G_train_loss.backward()
            optimizerG.step()

            if index % 20 == 0: print("D_loss:%f\tG_loss:%f" % (D_train_loss, G_train_loss))
            index += 1 * batchsize

main()
