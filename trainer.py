from model.reformercn import LitGPT
from pytorch_lightning.utilities.cli import LightningCLI 

cli = LightningCLI(LitGPT,save_config_overwrite=True)