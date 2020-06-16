import os
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from loadData import DataSet
import shutil
import random
import cv2

class generator(nn.Module):
    def __init__(self, config):
        super(generator, self).__init__()
        if config.category == "Mnist":
            self.fc1 = nn.Linear(config.z_size, 1024)
            self.bn1 = nn.BatchNorm1d(1024)

            self.fc2 = nn.Linear(1024, 1568)
            self.bn2 = nn.BatchNorm1d(1568)

            self.deconv1 = nn.ConvTranspose2d(1568//(7*7), 128, kernel_size=5, stride=2, padding=2, output_padding=1)
            self.bn3 = nn.BatchNorm2d(128)

            self.deconv2 = nn.ConvTranspose2d(128, 1, kernel_size=5, stride=2, padding=2, output_padding=1)

            self.leaky_relu = nn.LeakyReLU()
            self.tanh = nn.Tanh()

    def forward(self, z):
        out = self.fc1(z)
        out = self.bn1(out)
        out = self.leaky_relu(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.leaky_relu(out)
        out = out.view(out.size(0), -1, 7, 7)
        out = self.deconv1(out)
        out = self.bn3(out)
        out = self.leaky_relu(out)
        out = self.deconv2(out)
        out = self.tanh(out)
        return out



class Abp(nn.Module):
    def __init__(self, config):
        super(Abp, self).__init__()
        self.config = config
        self.prior = config.prior
        self.category = config.category
        self.epoch = config.Train_Epochs
        self.img_size = 28 if (config.category == 'Fashion-Mnist' or config.category == 'Mnist') else 64
        self.num = config.num
        self.batch_size = config.batch_size

        self.z_size = config.z_size
        self.langevin_num = config.langevin_num

        self.vis_step = config.vis_step

        self.lr = config.lr
        self.theta = config.theta
        self.delta = config.delta
        self.channel = 1 if (config.category == 'Fashion-Mnist' or config.category == 'Mnist') else 3

        self.checkpoint_dir = config.checkpoint_dir
        self.logs_dir = config.logs_dir
        self.recon_dir = config.recon_dir
        self.gen_dir = config.gen_dir


        self.with_noise = config.with_noise

        def create_directory(names):
            for i in names:
                if os.path.exists(i):
                    shutil.rmtree(i)
                os.makedirs(i)

        if config.isTraining == True:
            if config.continue_train == False:
                create_directory([self.logs_dir, self.recon_dir, self.gen_dir])


    def langevin_dynamic_generator(self, z, obs):
        obs = obs.detach()
        for i in range(self.langevin_num):
            z = Variable(z, requires_grad=True)
            gen = self.generator(z)
            loss = self.l2loss(gen, obs)
            loss.backward()
            grad = z.grad
            z = z - 0.5 * self.delta * self.delta * (grad + self.prior*z)
            if self.with_noise == True:
                noise = Variable(torch.normal(size=[self.batch_size, self.z_size], mean=0, std=1).cuda())
                z += self.delta * noise
        return z

    def l2loss(self, syn, obs):
        a = syn - obs
        return (1.0 / (2 * self.theta * self.theta) * torch.mul(a, a)).sum(dim=[1, 2, 3]).mean(dim=0)
        # return 1.0 / (2 * self.theta * self.theta) * loss(syn, obs)

    def train(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)
        self.generator = generator(self.config).to(device)

        data = DataSet(num=self.num, img_size=self.img_size, batch_size=self.batch_size, category=self.category)

        num_batches = int(len(data)/self.batch_size)

        optim = torch.optim.Adam(self.generator.parameters(), lr=self.lr)

        latents = torch.normal(size=[len(data), self.z_size], mean=0, std=1).cuda()

        for epoch in range(self.epoch):
            for i in range(num_batches):
                if (i + 1) * self.batch_size > len(data):
                    continue

                obs = data.NextBatch(i)
                obs = Variable(torch.Tensor(obs).cuda()).permute(0, 3, 1, 2)

                z = Variable(latents[i*self.batch_size: (i+1)*self.batch_size], requires_grad=True)

                z = self.langevin_dynamic_generator(z, obs)

                recon = self.generator(z)

                gen_loss = self.l2loss(recon, obs.detach())


                optim.zero_grad()
                gen_loss.backward()
                optim.step()
                latents[i*self.batch_size: (i+1)*self.batch_size] = z



            print(epoch, ": loss: ", gen_loss.data)
            if epoch % self.vis_step == 0:
                self.visualize(len(data), epoch, latents, data)

    def visualize(self, num_data, epoch, latent, data):

        idx = random.randint(0, int(num_data / self.batch_size) - 1)
        z = Variable(latent[idx * self.batch_size: (idx + 1) * self.batch_size].cuda())
        """
                Recon
        """
        obs = data.NextBatch(idx)
        obs = Variable(torch.Tensor(obs).cuda()).permute(0, 3, 1, 2)

        z = self.langevin_dynamic_generator(z, obs)
        sys = self.generator(z).permute(0,2,3,1).cpu().data
        sys = np.array((sys + 1) * 127.5, dtype=np.float)
        path = self.recon_dir + 'epoch' + str(epoch) + 'recon.jpg'
        # show_z_and_img(epoch, path, z, sys, self.row, self.col)
        self.show_in_one(path, sys, column=16, row=8)

        """
        Generation
        """
        # obs = data.NextBatch(idx, test=True)
        z = Variable(torch.normal(size=(self.batch_size, self.z_size), mean=0,std=1).cuda())
        # z = sess.run(self.langevin, feed_dict={self.z: z, self.x: obs})
        sys = self.generator(z).permute(0, 2, 3, 1).cpu().data
        sys = np.array((sys + 1) * 127.5, dtype=np.float)
        path = self.gen_dir + 'epoch' + str(epoch) + 'gens.jpg'
        self.show_in_one(path, sys, column=16, row=8)

    def show_in_one(self, path, images, column, row, show_size=[300, 300], blank_size=5):
        small_h, small_w = images[0].shape[:2]
        # column = int(show_size[1] / (small_w + blank_size))

        show_size[0] = small_h * row + row * blank_size
        show_size[1] = small_w * column + column * blank_size

        # row = int(show_size[0] / (small_h + blank_size))
        shape = [show_size[0], show_size[1]]
        for i in range(2, len(images[0].shape)):
            shape.append(images[0].shape[i])

        merge_img = np.zeros(tuple(shape), images[0].dtype)

        max_count = len(images)
        count = 0
        for i in range(row):
            if count >= max_count:
                break
            for j in range(column):
                if count < max_count:
                    im = images[count]
                    t_h_start = i * (small_h + blank_size)
                    t_w_start = j * (small_w + blank_size)
                    t_h_end = t_h_start + im.shape[0]
                    t_w_end = t_w_start + im.shape[1]

                    merge_img[t_h_start:t_h_end, t_w_start:t_w_end] = im
                    count = count + 1
                else:
                    break
        cv2.imwrite(path, merge_img)
        # cv2.namedWindow(window_name)
        # cv2.imshow(window_name, merge_img)



























