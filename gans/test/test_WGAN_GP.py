""" This tests the generator function. """
import unittest

import torch

from gans.models.WGAN_GP.generator import generator, get_noise
from gans.models.WGAN_GP.critic import critic
from gans.models.WGAN_GP.gradient_penalty import get_gradient, gradient_penalty


class test_get_penalty(unittest.TestCase):
    """
    Test get_gradient.py
    """
    def test_get_gradient(self):
        """
        Test get_gradient()
        """
        image_shape = (256, 1, 28, 28)
        real = torch.randn(*image_shape) + 1
        fake = torch.randn(*image_shape) - 1
        epsilon_shape = [1 for _ in image_shape]
        epsilon_shape[0] = image_shape[0]
        epsilon = torch.rand(epsilon_shape).requires_grad_()
        gradient = get_gradient(critic, real, fake, epsilon)

        self.assertTrue(tuple(gradient.shape) == image_shape)
        self.assertGreater(gradient.max(), 0)
        self.assertLess(gradient.min(), 0)


class test_gradient_penalty(unittest.TestCase):
    """
    Test gradient_penalty.py
    """
    def test_gradient_penalty(self):
        
        image_shape = (256, 1, 28, 28)
        
        bad_gradient = torch.zeros(*image_shape)
        bad_gradient_penalty = gradient_penalty(bad_gradient)
        self.assertAlmostEqual(bad_gradient_penalty.item(), 1)

        image_size = torch.prod(torch.Tensor(image_shape[1:]))
        good_gradient = torch.ones(*image_shape) / torch.sqrt(image_size)
        good_gradient_penalty = gradient_penalty(good_gradient)
        self.assertAlmostEqual(good_gradient_penalty.item(), 0.)

        random_gradient = test_get_gradient(image_shape)
        random_gradient_penalty = gradient_penalty(random_gradient)
        self.assertLess(torch.abs(random_gradient_penalty - 1), 0.1)


if __name__ == '__main__':
    unittest.main()