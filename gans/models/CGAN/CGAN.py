import torch
import torchvision
from torch import nn
from pytorch_lightning import LightningModule
import torch.nn.functional as F

from ..DCGAN.generator import generator, get_noise
from ..DCGAN.discriminator import discriminator

class CGAN(LightningModule):
    '''
    DCGAN Class
    Values:
        hidden_dim: the dimension of the noise vector, a scalar
        im_dim: the dimension of the images, fitted for the dataset used, a scalar
          (MNIST images are 28 x 28 = 784 so that is your default)
        hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self,
    im_chan: int = 1,
    z_dim: int = 10,
    hidden_dim: int = 64,
    lr: float = 0.0002,
    beta_1: float = 0.5,
    beta_2: float = 0.999,
    spectral_norm: bool = True,
    n_classes: int = 10,
    **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.generator = generator(z_dim=self.hparams.z_dim + n_classes, hidden_dim=self.hparams.hidden_dim, im_chan=self.hparams.im_chan).apply(self.weights_init)
        self.discriminator = discriminator(im_chan=self.hparams.im_chan + n_classes, hidden_dim=self.hparams.hidden_dim, spectral_norm=spectral_norm).apply(self.weights_init)
    
    def weights_init(self, m):
        """initialize the weights to the normal distribution with mean 0 and standard deviation 0.02

        Args:
            m (nn.Module): nn.Module, generator or discriminator
        """
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
        if isinstance(m, nn.BatchNorm2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
            torch.nn.init.constant_(m.bias, 0)

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        criterion = nn.BCEWithLogitsLoss()
        return criterion(y_hat, y)

    def training_step(self, train_batch, batch_idx, optimizer_idx):

        imgs, labels = train_batch
        one_hot_labels = self.get_one_hot_labels(labels)
        image_one_hot_labels = one_hot_labels[:, :, None, None]
        image_one_hot_labels = image_one_hot_labels.repeat(1, 1, imgs.shape[2], imgs.shape[3])

        real_image_and_labels = self.combine_vectors(imgs, image_one_hot_labels)

        # train generator
        if optimizer_idx == 0:
            

            # put on GPU because we created this tensor inside training_loop
            # sample noise
            fake_noise = get_noise(len(imgs), self.hparams.z_dim)
            fake_noise = fake_noise.type_as(imgs)
            # generate images
            noise_and_labels = self.combine_vectors(fake_noise, one_hot_labels)
            fake = self.generator(noise_and_labels)

            fake_image_and_labels = self.combine_vectors(fake, image_one_hot_labels)
            disc_fake_pred = self.discriminator(fake_image_and_labels)
            g_loss = self.adversarial_loss(disc_fake_pred, torch.ones_like(disc_fake_pred))
            
            self.log("loss/g_loss", g_loss, prog_bar=True)
            
            return g_loss

        # train discriminator
        if optimizer_idx == 1:
            # Measure discriminator's ability to classify real from generated samples
            
            # sample noise
            fake_noise2 = get_noise(len(imgs), self.hparams.z_dim)
            fake_noise2 = fake_noise2.type_as(imgs)
            noise_and_labels = self.combine_vectors(fake_noise2, one_hot_labels)
            fake = self.generator(noise_and_labels)

            fake_image_and_labels = self.combine_vectors(fake, image_one_hot_labels)
            disc_fake_pred = self.discriminator(fake_image_and_labels)
            disc_real_pred = self.discriminator(real_image_and_labels)

            fake_loss = self.adversarial_loss(disc_fake_pred, torch.zeros_like(disc_fake_pred))
            real_loss = self.adversarial_loss(disc_real_pred, torch.ones_like(disc_real_pred))

            # discriminator loss is the average of these
            d_loss = (real_loss + fake_loss) / 2
            self.log("loss/d_loss", d_loss, prog_bar=True)
            return d_loss

    def configure_optimizers(self):
        gen_opt = torch.optim.Adam(self.generator.parameters(), lr=self.hparams.lr, betas=(self.hparams.beta_1, self.hparams.beta_2))
        disc_opt = torch.optim.Adam(self.discriminator.parameters(), lr=self.hparams.lr, betas=(self.hparams.beta_1, self.hparams.beta_2))
        return gen_opt, disc_opt
    
    # def validation_step(self, batch, batch_idx):
    #     pass

    def on_validation_epoch_end(self, n_samples=25, labels=None):
        
        z = get_noise(n_samples=n_samples, z_dim=self.hparams.z_dim)

        if type(labels) == list:
            labels = torch.tensor(labels)
        elif type(labels) in [int, str]:
            labels = torch.tensor(int(labels)).repeat(n_samples)
        elif not labels:
            labels = torch.randint(0,10,(n_samples,))

        one_hot_labels = self.get_one_hot_labels(labels)
        noise_and_labels = self.combine_vectors(z, one_hot_labels)

        noise_and_labels = noise_and_labels.type_as(self.generator.generator[0][0].weight)

        # log sampled images
        sample_imgs = self(noise_and_labels)

        image_unflat = sample_imgs.detach().cpu().view(-1, 1, 28, 28)
        grid = torchvision.utils.make_grid(image_unflat, nrow=5)
        self.logger.experiment.add_image("generated_images", grid, self.current_epoch)
        return grid # sample_imgs# 

    def get_one_hot_labels(self, labels):
        '''
        Function for creating one-hot vectors for the labels, returns a tensor of shape (?, num_classes).
        Parameters:
            labels: tensor of labels from the dataloader, size (?)
            self.hparams.n_classes: the total number of classes in the dataset, an integer scalar
        '''
        return F.one_hot(labels, num_classes=self.hparams.n_classes)
    
    def combine_vectors(self, x, y):
        '''
        Function for combining two vectors with shapes (n_samples, ?) and (n_samples, ?).
        Parameters:
        x: (n_samples, ?) the first vector. 
            In this assignment, this will be the noise vector of shape (n_samples, z_dim), 
            but you shouldn't need to know the second dimension's size.
        y: (n_samples, ?) the second vector.
            Once again, in this assignment this will be the one-hot class vector 
            with the shape (n_samples, self.hparams.n_classes), but you shouldn't assume this in your code.
        '''
        combined = torch.cat((x.to(torch.float), y.to(torch.float)), 1)
        return combined

    def get_input_dimensions(self, mnist_shape):
        '''
        Function for getting the size of the conditional input dimensions 
        from z_dim, the image shape, and number of classes.
        Parameters:
            z_dim: the dimension of the noise vector, a scalar
            mnist_shape: the shape of each MNIST image as (C, W, H), which is (1, 28, 28)
            self.hparams.n_classes: the total number of classes in the dataset, an integer scalar
                    (10 for MNIST)
        Returns: 
            generator_input_dim: the input dimensionality of the conditional generator,
                            which takes the noise and class vectors
            discriminator_im_chan: the number of input channels to the discriminator
                                (e.g. C x 28 x 28 for MNIST)
        '''
        generator_input_dim = self.hparams.z_dim + self.hparams.n_classes
        discriminator_im_chan = mnist_shape[0] + self.hparams.n_classes
        return generator_input_dim, discriminator_im_chan