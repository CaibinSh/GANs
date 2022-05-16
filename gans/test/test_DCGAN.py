""" This tests the generator function. """
import unittest
import numpy as np
import torch
from torch import nn

from gans.models.DCGAN.generator import generator, get_noise
from gans.models.DCGAN.discriminator import discriminator


class generator_test(unittest.TestCase):
    """
    Test generator.py
    """

    def test_generator(self):
        """
        Test make_generator_block()
        """
        num_test = 100
        
        gen = generator()

        # Test the hidden block
        test_hidden_noise = get_noise(num_test, gen.z_dim)
        test_hidden_block = gen.make_generator_block(10, 20, kernel_size=4, stride=1)
        test_uns_noise = gen.unsqueeze_noise(test_hidden_noise)
        hidden_output = test_hidden_block(test_uns_noise)

        # Check that it works with other strides
        test_hidden_block_stride = gen.make_generator_block(20, 20, kernel_size=4, stride=2)

        test_final_noise = get_noise(num_test, gen.z_dim) * 20
        test_final_block = gen.make_generator_block(10, 20, final_layer=True)
        test_final_uns_noise = gen.unsqueeze_noise(test_final_noise)
        final_output = test_final_block(test_final_uns_noise)

        # Test the whole thing:
        test_gen_noise = get_noise(num_test, gen.z_dim)
        test_uns_gen_noise = gen.unsqueeze_noise(test_gen_noise)
        gen_output = gen(test_uns_gen_noise)

        self.assertTrue(tuple(hidden_output.shape) == (num_test, 20, 4, 4), "hidden output dimensions not right") 
        self.assertGreater(hidden_output.max(), 1)
        self.assertTrue(hidden_output.min() == 0)
        self.assertGreater(hidden_output.std(), 0.2)
        self.assertLess(hidden_output.std(), 1)
        self.assertGreater(hidden_output.std(), 0.5)

        self.assertTrue(tuple(test_hidden_block_stride(hidden_output).shape) == (num_test, 20, 10, 10))

        self.assertTrue(final_output.max().item() == 1)
        self.assertTrue(final_output.min().item() == -1)

        self.assertTrue(tuple(gen_output.shape) == (num_test, 1, 28, 28))
        self.assertGreater(gen_output.std(), 0.5)
        self.assertLess(gen_output.std(), 0.8)
        
class discriminator_test(unittest.TestCase):
    """
    Test generator.py
    """

    def test_discriminator_block(self):
        '''
        Test your make_discriminator_block() function
        ''' 
        num_test = 100

        gen = generator()
        disc = discriminator()

        test_images = gen(get_noise(num_test, gen.z_dim))

        # Test the hidden block
        test_hidden_block = disc.make_discriminator_block(1, 5, kernel_size=6, stride=3)
        hidden_output = test_hidden_block(test_images)

        # Test the final block
        test_final_block = disc.make_discriminator_block(1, 10, kernel_size=2, stride=5, final_layer=True)
        final_output = test_final_block(test_images)

        # Test the whole thing:
        disc_output = disc(test_images)

        # Test the hidden block
        self.assertTrue(tuple(hidden_output.shape) == (num_test, 5, 8, 8))
        # Because of the LeakyReLU slope
        self.assertGreater(-hidden_output.min() / hidden_output.max(), 0.15)
        self.assertLess(-hidden_output.min() / hidden_output.max(), 0.25)
        self.assertGreater(hidden_output.std(), 0.5)
        self.assertLess(hidden_output.std(), 1)

        # Test the final block

        self.assertTrue(tuple(final_output.shape) == (num_test, 10, 6, 6))
        self.assertGreater(final_output.max(), 1.0)
        self.assertLess(final_output.min(), -1.0)
        self.assertGreater(final_output.std(), 0.3)
        self.assertLess(final_output.std(), 0.6)

        # Test the whole thing:

        self.assertTrue(tuple(disc_output.shape) == (num_test, 1))
        self.assertGreater(disc_output.std(), 0.25)
        self.assertLess(disc_output.std(), 0.5)


if __name__ == '__main__':
    unittest.main()