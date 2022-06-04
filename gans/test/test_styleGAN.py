""" This tests the generator function. """
from tkinter.filedialog import test
import unittest

import torch
from torch import nn

from gans.models.styleGAN import (
    MappingLayers, InjectNoise, AdaIN, MicroStyleGANGeneratorBlock, MicroStyleGANGenerator, get_truncated_noise
)

class test_MappingLayers(unittest.TestCase):
    """
    Test MappingLayers.py
    """
    def test_mappinglayers(self):
        """
        Test MappingLayers class
        """

        map_fn = MappingLayers(z_dim=10, hidden_dim=20, w_dim=30)
        self.assertTrue(tuple(map_fn(torch.randn(2,10)).shape) == (2, 30))
        self.assertGreater(len(map_fn.mapping), 4)

        outputs = map_fn(torch.randn(1000, 10))
        self.assertGreater(outputs.std(), 0.03)
        self.assertLess(outputs.std(), 0.3)
        self.assertGreater(outputs.min(), -2)
        self.assertLess(outputs.min(), 0)
        self.assertGreater(outputs.max(), 0)
        self.assertLess(outputs.max(), 2)

        layers = [str(x).replace(" ", "").replace("inplace=True", "") for x in map_fn.get_mapping()]
        expected = ['Linear(in_features=10,out_features=20,bias=True)', 
                  'ReLU()',
                  'Linear(in_features=20,out_features=20,bias=True)', 
                  'ReLU()', 
                  'Linear(in_features=20,out_features=30,bias=True)']
        self.assertEqual(layers, expected)

    def test_injectnoise(self):
        """
        Test InjectNoise class
        """

        test_noise_channels = 3000
        test_noise_samples = 20
        fake_images = torch.randn(test_noise_samples, test_noise_channels, 10, 10)
        inject_noise = InjectNoise(test_noise_channels)

        self.assertLess(torch.abs(inject_noise.weight.std()-1), 0.1)
        self.assertLess(torch.abs(inject_noise.weight.mean()), 0.1)
        self.assertEqual(type(inject_noise.get_weight()), torch.nn.parameter.Parameter)

        # check that something changed
        inject_noise.weight = nn.Parameter(torch.ones_like(inject_noise.weight))
        self.assertGreater(torch.abs(inject_noise(fake_images) - fake_images).mean(), 0.1)

        # check that the change is per channel
        self.assertGreater(torch.abs((inject_noise(fake_images) - fake_images).std(0)).mean(), 1e-4)
        self.assertLess(torch.abs((inject_noise(fake_images) - fake_images).std(1)).mean(), 1e-4)
        self.assertGreater(torch.abs((inject_noise(fake_images) - fake_images).std(2)).mean(), 1e-4)
        self.assertGreater(torch.abs((inject_noise(fake_images) - fake_images).std(3)).mean(), 1e-4)

        # check that the per-channel change is roughly normal
        per_channel_change = (inject_noise(fake_images) - fake_images).mean(1).std()
        self.assertGreater(per_channel_change, .9)
        self.assertLess(per_channel_change, 1.1)

        # ensure that the weights are being used at all
        inject_noise.weight = nn.Parameter(torch.zeros_like(inject_noise.weight))
        self.assertLess(torch.abs(inject_noise(fake_images) - fake_images).mean(), 1e-4)
        self.assertEqual(len(inject_noise.weight.shape), 4)

    def test_adain(self):
        """
        Test AdaIN class
        """

        w_channels = 50
        image_channels = 20
        image_size = 30
        n_test = 10

        adain = AdaIN(image_channels, w_channels)
        test_w = torch.randn(n_test, w_channels)

        self.assertEqual(adain.style_scale_transform(test_w).shape, adain.style_shift_transform(test_w).shape)
        self.assertEqual(adain.style_scale_transform(test_w).shape[-1], image_channels)
        self.assertEqual(
            tuple(
                adain(torch.randn(n_test, image_channels, image_size, image_size), test_w).shape
            ),
            (n_test, image_channels, image_size, image_size)
        )

        w_channels = 3
        image_channels = 2
        image_size = 3
        n_test = 1
        adain = AdaIN(image_channels, w_channels)

        adain.style_scale_transform.weight.data = torch.ones_like(adain.style_scale_transform.weight.data) / 4
        adain.style_scale_transform.bias.data = torch.zeros_like(adain.style_scale_transform.bias.data)
        adain.style_shift_transform.weight.data = torch.ones_like(adain.style_shift_transform.weight.data) / 5
        adain.style_shift_transform.bias.data = torch.zeros_like(adain.style_shift_transform.bias.data)

        test_input = torch.ones(n_test, image_channels, image_size, image_size)
        test_input[:, :, 0] = 0 
        test_w = torch.ones(n_test, w_channels)
        test_output = adain(test_input, test_w)
        self.assertLess(torch.abs(test_output[0, 0, 0, 0] - 3 / 5 + torch.sqrt(torch.tensor(9 / 8))), 1e-4)
        self.assertLess(torch.abs(test_output[0, 0, 1, 0] - 3 / 5 - torch.sqrt(torch.tensor(9 / 32))), 1e-4)

    def test_microstylegangeneratorblock(self):
        """
        Test MicroStyleGANGenertorBlock class
        """

        test_stylegan_block = MicroStyleGANGeneratorBlock(in_chan=128, out_chan=64, w_dim=256, kernel_size=3, starting_size=8)
        test_x = torch.ones(1, 128, 4, 4)
        test_x[:, :, 1:3, 1:3] = 0
        test_w = torch.ones(1, 256)
        test_x = test_stylegan_block.upsample(test_x)

        self.assertEqual(tuple(test_x.shape), (1, 128, 8, 8))
        self.assertLess(torch.abs(test_x.mean() - 0.75), 1e-4)

        test_x = test_stylegan_block.conv(test_x)
        self.assertEqual(tuple(test_x.shape), (1, 64, 8, 8))

        test_x = test_stylegan_block.inject_noise(test_x)
        test_x = test_stylegan_block.activation(test_x)
        self.assertLess(test_x.min(), 0)
        self.assertLess(-test_x.min() / test_x.max(), 0.4)
    
    def test_microstylegangenerator(self):
        z_dim = 128
        out_chan = 3
        truncation = 0.7

        mu_stylegan = MicroStyleGANGenerator(
            z_dim=z_dim, 
            map_hidden_dim=1024,
            w_dim=496,
            in_chan=512,
            out_chan=out_chan, 
            kernel_size=3, 
            hidden_chan=256
        )

        test_samples = 10
        test_result = mu_stylegan(get_truncated_noise(test_samples, z_dim, truncation))
        self.assertEqual(tuple(test_result.shape), (test_samples, out_chan, 16, 16))
        
        # chech that the interpolation is correct
        for alpha in [0., 1.]:
            mu_stylegan.alpha = alpha
            test_result, _, test_big = mu_stylegan(
                get_truncated_noise(test_samples, z_dim, truncation), return_intermediate=True
            )
        self.assertLess(torch.abs(test_result - test_big).mean(), 0.001)

if __name__ == '__main__':
    unittest.main()