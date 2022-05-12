""" This tests the generator function. """
from dis import disco
from numbers import Real
import unittest
import numpy as np
import torch
from torch import nn

from gans.generator import generator
from gans.discriminator import discriminator
from gans.loss_function import get_disc_loss, get_gen_loss
from gans.myGANs import MNISTDataModule

class loss_test(unittest.TestCase):
    """
    Test loss_function.py
    """

    def rationale_disc_loss(self):
        """
        Test rationale of disc loss
        """
        z_dim, num_images = 64, 10
        gen = torch.zeros_like
        disc = lambda x: x.mean(1)[:, None]
        criterion = torch.mul
        real = torch.ones(num_images, z_dim)
        disc_loss = get_disc_loss(gen, disc, criterion, real, num_images, z_dim, device="cpu")
        self.assertLess(torch.abs(disc_loss.mean()) - 0.5, 1e-5)
        
        gen = torch.ones_like
        real = torch.zeros(num_images, z_dim)
        disc_loss = get_disc_loss(gen, disc, criterion, real, num_images, z_dim, device="cpu")
        self.assertLess(torch.abs(disc_loss.mean()), 1e-5)


    def test_disc_loss(self):
        """
        Test disc loss()
        """
        max_tests = 10
        z_dim = 64
        criterion = nn.BCEWithLogitsLoss()
        gen = generator(z_dim).to(device="cpu")
        disc = discriminator().to(device="cpu") 
        disc_opt = torch.optim.Adam(disc.parameters(), lr=0.00001)
        num_steps = 0
        mydata = MNISTDataModule().train_dataloader()
        for real, _ in mydata:
            cur_batch_size = len(real)
            real = real.view(cur_batch_size, -1).to("cpu")

            ### Update discriminator ###
            # Zero out the gradient before backpropagation
            disc_opt.zero_grad()

            # Calculate discriminator loss
            disc_loss = get_disc_loss(gen, disc, criterion, real, cur_batch_size, z_dim, "cpu")
            self.assertLess((disc_loss - 0.68).abs(), 0.05)

            # Update gradients
            disc_loss.backward(retain_graph=True)

            # Check that they detached correctly
            self.assertTrue(gen.gen[0][0].weight.grad is None)

            # Update optimizer
            old_weight = disc.disc[0][0].weight.data.clone()
            disc_opt.step()
            new_weight = disc.disc[0][0].weight.data
            
            # Check that some discriminator weights changed
            self.assertTrue(not torch.all(torch.eq(old_weight, new_weight)))
            num_steps += 1
            if num_steps >= max_tests:
                break

    def rationale_gen_loss(self):
        """
        Test rationale of gen loss
        """
        z_dim, num_images = 64, 10
        gen = torch.zeros_like
        disc = nn.Identity()
        criterion = torch.mul
        real = torch.ones(num_images, z_dim)
        gen_loss_tensor = get_gen_loss(gen, disc, criterion, num_images, z_dim, "cpu")
        self.assertLess(torch.abs(gen_loss_tensor), 1e-5)
        
        gen = torch.ones_like
        real = torch.zeros(num_images, 1)
        gen_loss_tensor = get_gen_loss(gen, disc, criterion, num_images, z_dim, "cpu")
        self.assertLess(torch.abs(disc_loss.mean()) - 1, 1e-5)


    def test_gen_loss(self):
        """
        Test disc loss()
        """
        max_tests = 10
        z_dim = 64
        criterion = nn.BCEWithLogitsLoss()
        gen = generator(z_dim).to(device="cpu")
        gen_opt = torch.optim.Adam(gen.parameters(), lr=0.00001)
        disc = discriminator().to(device="cpu") 
        disc_opt = torch.optim.Adam(disc.parameters(), lr=0.00001)
        num_steps = 0
        mydata = MNISTDataModule().train_dataloader()
        for real, _ in mydata:
            cur_batch_size = len(real)
            real = real.view(cur_batch_size, -1).to("cpu")

            ### Update discriminator ###
            # Zero out the gradient before backpropagation
            disc_opt.zero_grad()

            # Calculate discriminator loss
            disc_loss = get_disc_loss(gen, disc, criterion, real, cur_batch_size, z_dim, "cpu")
            self.assertLess((disc_loss - 0.68).abs(), 0.05)

            # Update gradients
            disc_loss.backward(retain_graph=True)

            # Check that they detached correctly
            self.assertTrue(gen.gen[0][0].weight.grad is None)

            # Update optimizer
            old_weight = disc.disc[0][0].weight.data.clone()
            disc_opt.step()
            new_weight = disc.disc[0][0].weight.data
            
            # Check that some discriminator weights changed
            self.assertTrue(not torch.all(torch.eq(old_weight, new_weight)))
            num_steps += 1
            if num_steps >= max_tests:
                break
                
if __name__ == '__main__':
    unittest.main()