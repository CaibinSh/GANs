from .discriminator import discriminator
from .generator import generator, get_noise

def get_disc_loss(gen, disc, criterion, real, num_images, z_dim, device=):
    '''
    Return the loss of the discriminator given inputs.
    Parameters:
        gen: the generator model, which returns an image given z-dimensional noise
        disc: the discriminator model, which returns a single-dimensional prediction of real/fake
        criterion: the loss function, which should be used to compare 
               the discriminator's predictions to the ground truth reality of the images 
               (e.g. fake = 0, real = 1)
        real: a batch of real images
        num_images: the number of images the generator should produce, 
                which is also the length of the real images
        z_dim: the dimension of the noise vector, a scalar
        device: the device type
    Returns:
        disc_loss: a torch scalar loss value for the current batch
    '''
    noise = get_noise(num_images, z_dim, device=device)
    fake_images = gen(noise).detach()
    pred = disc(fake_images)
    loss_fake = criterion(pred, torch.zeros_like(pred))
    ground_truth = disc(real)
    loss_real = criterion(ground_truth, torch.ones_like(ground_truth))
    disc_loss = (loss_fake + loss_real) / 2
    return disc_loss
