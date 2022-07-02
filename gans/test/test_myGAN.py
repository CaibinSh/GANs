""" This tests the generator function. """
import unittest
import numpy as np
import torch
from torch import nn

from gans.models.firstGAN.generator import get_generator_block, generator
from gans.models.firstGAN.discriminator import get_discriminator_block, discriminator

class test_discriminator(unittest.TestCase):
    """
    Test generator.py
    """

    def test_discriminatior_block(self):
        """
        Test discriminatior()
        """
        in_features, out_features, num_test=25, 12, 10000

        dis_block = get_discriminator_block(in_features, out_features)
        
        # Check there are two parts
        self.assertTrue(len(dis_block)==2, "block number is not 2")
        self.assertIsInstance(dis_block[0], nn.Linear, "1st block is not Linear")
        self.assertIsInstance(dis_block[1], nn.LeakyReLU, "2nd block is not LeakyReLU")
        
        test_input = torch.randn(num_test, in_features)
        test_output = dis_block(test_input)
        
        # Check that the shape is right
        self.assertTrue(np.array_equal(test_output.shape, (num_test, out_features)))

        # Check that the LeakyReLU slope is about 0.2
        self.assertGreater(-test_output.min() / test_output.max(), 0.1, "LeakyReLU slope probably is not 0.2")
        self.assertLess(-test_output.min() / test_output.max(), 0.3, "LeakyReLU slope probably is not 0.2")
        self.assertGreater(test_output.std(), 0.3, "LeakyReLU slope probably is not 0.2")
        self.assertLess(test_output.std(), 0.5, "LeakyReLU slope probably is not 0.2")
    
    # Verify the discriminator class    
    def test_discriminatior(self):
        """
        Test discriminatior()
        """

        img_shape, num_tests = (1, 28, 28), 100

        disc = discriminator(img_shape).get_disc()

        # Check there are five parts
        self.assertTrue(len(disc)==7, "block number is not 7")

        test_input = torch.randn(num_tests, *img_shape)
        test_flat = test_input.view(test_input.size(0), -1)
        test_output = disc(test_flat)
        
        self.assertTrue(np.array_equal(test_output.shape, (num_tests, 1)))

class test_generator(unittest.TestCase):
    """
    Test generator.py
    """

    def test_generator_block(self):
        """
        Test generator()
        """
        in_features, out_features, num_test=40, 20, 1000

        gen_block = get_generator_block(in_features, out_features)

        self.assertTrue(len(gen_block)==3, "block number is not 3")
        self.assertIsInstance(gen_block[0], nn.Linear, "1st block is not Linear")
        self.assertIsInstance(gen_block[1], nn.BatchNorm1d, "2nd block is not BatchNorm")
        self.assertIsInstance(gen_block[2], nn.ReLU, "3rd block is not ReLU")
        
        test_input = torch.randn(num_test, in_features)
        test_output = gen_block(test_input)
        self.assertTrue(np.array_equal(test_output.shape, (num_test, out_features)))

        self.assertGreater(test_output.std(), .55)
        self.assertLess(test_output.std(), .65)
    
    def test_generator(self):
        """
        Test generator()
        """

        latent_dim, img_shape, z_dim, num_test = 10, (1, 28, 28), 10, 10000

        gen = generator(z_dim, latent_dim, img_shape).get_gen()

        self.assertTrue(len(gen)==14, "block number is not 14")

        test_input = torch.randn(num_test, latent_dim)
        test_output = gen(test_input)

        self.assertTrue(np.array_equal(test_output.shape, (num_test, int(np.prod(img_shape)))), "output dimension not expected")
        
        self.assertLess(test_output.min(), .5, "Don't use a block in your solution")
        self.assertGreater(test_output.std(), .05, "Don't use batchnorm here")
        self.assertLess(test_output.std(), .15, "Don't use batchnorm here")
        

if __name__ == '__main__':
    unittest.main()