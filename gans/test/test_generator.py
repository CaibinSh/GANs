""" This tests the generator function. """
import unittest
import numpy as np
import torch
from torch import nn

from gans.generator import get_generator_block, generator

class generator_test(unittest.TestCase):
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

        z_dim, im_dim, hidden_dim, num_test = 5, 10, 20, 10000

        gen = generator(z_dim, im_dim, hidden_dim).get_gen()

        self.assertTrue(len(gen)==6, "block number is not 6")

        test_input = torch.randn(num_test, z_dim)
        test_output = gen(test_input)

        self.assertTrue(np.array_equal(test_output.shape, (num_test, im_dim)))
        
        self.assertLess(test_output.min(), .5, "Don't use a block in your solution")
        self.assertGreater(test_output.std(), .05, "Don't use batchnorm here")
        self.assertLess(test_output.std(), .15, "Don't use batchnorm here")
        
  
if __name__ == '__main__':
    unittest.main()