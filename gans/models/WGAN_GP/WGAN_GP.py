import torch
from torch import nn
from pytorch_lightning import LightningModule
import torchvision

from .generator import generator, get_noise
from .critic import critic
from .gradient_penalty import one_L_enforcement as gp

class WGAN_GP(LightningModule):
    '''
    WGAN_GP Class
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
    n_epochs: int = 100,
    c_lambda: float= 10,
    crit_repeats: int = 5,
    **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.generator = generator(z_dim=self.hparams.z_dim, hidden_dim=self.hparams.hidden_dim, im_chan=self.hparams.im_chan).apply(self.weights_init)
        self.critic = critic(im_chan=self.hparams.im_chan, hidden_dim=self.hparams.hidden_dim).apply(self.weights_init)
    
    def weights_init(self, m):
        """initialize the weights to the normal distribution with mean 0 and standard deviation 0.02

        Args:
            m (nn.Module): nn.Module, generator or critic
        """
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
        if isinstance(m, nn.BatchNorm2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
            torch.nn.init.constant_(m.bias, 0)

    def forward(self, z):
        return self.generator(z)

    def training_step(self, train_batch, batch_idx, optimizer_idx):

        imgs, _ = train_batch

        # train generator
        if optimizer_idx == 0:
            
            g_loss = self.get_gen_loss(imgs)
            self.log("loss/g_loss", g_loss, prog_bar=True)
            
            return g_loss

        # train critic
        if optimizer_idx == 1:
            
            # critic loss is the average of these
            d_loss = self.get_crit_loss(imgs)

            self.log("loss/d_loss", d_loss, prog_bar=True)
            
            return d_loss

    def configure_optimizers(self):
        gen_opt = torch.optim.Adam(self.generator.parameters(), lr=self.hparams.lr, betas=(self.hparams.beta_1, self.hparams.beta_2))
        disc_opt = torch.optim.Adam(self.critic.parameters(), lr=self.hparams.lr, betas=(self.hparams.beta_1, self.hparams.beta_2))
        return gen_opt, disc_opt
    
    def get_gen_loss(self, imgs):
        '''
        Return the loss of a generator given the critic's scores of the generator's fake images.
        Parameters:
            crit_fake_pred: the critic's scores of the fake images
        Returns:
            gen_loss: a scalar loss value for the current batch of the generator
        '''

        fake_noise = get_noise(len(imgs), self.hparams.z_dim)
        fake_noise = fake_noise.type_as(imgs)

        # generate images
        self.generated_imgs = self(fake_noise)

        # adversarial loss is binary cross-entropy
        crit_fake_pred = self.critic(self.generated_imgs)
        gen_loss =  - crit_fake_pred.mean()
        return gen_loss

    def get_crit_loss(self, imgs, c_lambda=10, crit_repeats=5):
        '''
        Return the loss of a critic given the critic's scores for fake and real images,
        the gradient penalty, and gradient penalty weight.
        Parameters:
            crit_fake_pred: the critic's scores of the fake images
            crit_real_pred: the critic's scores of the real images
            gp: the unweighted gradient penalty
            c_lambda: the current weight of the gradient penalty
        Returns:
            crit_loss: a scalar for the critic's loss, accounting for the relevant factors
        '''
        critic_losses = torch.zeros(1)
        critic_losses = critic_losses.type_as(imgs)

        for _ in range(crit_repeats):

            fake_noise = get_noise(len(imgs), self.hparams.z_dim)
            fake_noise = fake_noise.type_as(imgs)
            fake = self(fake_noise)
            crit_fake_pred = self.critic(fake)
            crit_real_pred = self.critic(imgs)

            epsilon = torch.rand(len(imgs), 1, 1, 1, requires_grad=True)
            epsilon = epsilon.type_as(imgs)
 
            reg = gp(self.critic, imgs, fake, epsilon, c_lambda)
            crit_loss = - crit_real_pred.mean() + crit_fake_pred.mean() + reg
            critic_losses += crit_loss / crit_repeats

        return critic_losses

    def validation_step(self, batch, batch_idx):
        pass

    def on_validation_epoch_end(self):
        
        z = get_noise(n_samples=25, z_dim=self.hparams.z_dim)
        z = z.type_as(self.generator.generator[0][0].weight)

        # log sampled images
        sample_imgs = self(z)

        image_unflat = sample_imgs.detach().cpu().view(-1, 1, 28, 28)
        grid = torchvision.utils.make_grid(image_unflat, nrow=5)
        self.logger.experiment.add_image("generated_images", grid, self.current_epoch)
        return grid
