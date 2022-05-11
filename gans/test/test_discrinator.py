""" This tests the generator function. """
import sys
from pathlib import Path
import unittest
import numpy as np
import torch
from torch import nn

from gans.discriminator import get_discriminator_block, discriminator

class discriminator_test(unittest.TestCase):
    """
    Test generator.py
    """

    def test_discriminatior_block(self):
        """
        Test discriminatior()
        """
        in_features, out_features, num_test=40, 20, 1000

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
        self.assertGreater(test_output.std(), 0.5, "LeakyReLU slope probably is not 0.2")
    
    # Verify the discriminator class    
    def test_discriminatior(self):
        """
        Test discriminatior()
        """

        z_dim, hidden_dim, num_test = 20, 8, 100

        disc = discriminator(z_dim, hidden_dim).get_disc()

        # Check there are three parts
        self.assertTrue(len(disc)==4, "block number is not 4")

        test_input = torch.randn(num_test, z_dim)
        test_output = disc(test_input)

        self.assertTrue(np.array_equal(test_output.shape, (num_test, 1)))

if __name__ == '__main__':
    unittest.main()