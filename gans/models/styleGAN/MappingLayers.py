from torch import nn
from pytorch_lightning import LightningModule

class MappingLayers(LightningModule):
    """Mapping Layers Class mapping the noise vector z to an intermediate noise vector w
    """

    def __init__(self, z_dim, hidden_dim, w_dim) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.mapping = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, w_dim)
        )

    def forward(self, noise):
        return self.mapping(noise)

    # UNIT TEST
    def get_mapping(self):
        return self.mapping