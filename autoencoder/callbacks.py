import lightning as L
import matplotlib.pyplot as plt
import torch
import wandb
from lightning.pytorch.loggers import WandbLogger
from utility import Utility
from datasettype import DatasetType


class CustomCallbacks(L.Callback):
    def __init__(self, plot_ever_n_epoch, z_dim, wandb_logger: WandbLogger, dataset: DatasetType):
        super().__init__()
        self.plot_ever_n_epoch = plot_ever_n_epoch
        self.z_dim = z_dim
        self.wandb_logger = wandb_logger
        self.dataset = dataset

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module):
        # Every 10th epoch, generate some images
        if trainer.current_epoch % self.plot_ever_n_epoch == 0:
            gen_z = torch.randn((100, self.z_dim), requires_grad=False, device=pl_module.device)

            samples = pl_module.decoder(gen_z)
            samples = samples.permute(0, 2, 3, 1).contiguous().cpu().data.numpy()
            im = Utility.convert_to_display(samples)
            plt.imshow(im, cmap='Greys_r')

            # store in log folder (wandb log folder)
            wandb_folder = wandb.run.dir
            plt.savefig(f"{wandb_folder}/epoch_{trainer.current_epoch}.png")

            # log to wandb
            self.wandb_logger.log_image("samples_gaussian", images=[im], step=trainer.global_step)

    # when training is done
    def on_train_end(self, trainer, pl_module):
        # encode and decode some images
        data_module, im_shape, project_name = Utility.setup(dataset=self.dataset, batch_size=200, num_workers=1)
        data_module.prepare_data()
        data_module.setup("test")
        for x, y in data_module.test_dataloader():
            x = x.to(pl_module.device)
            x = x[:100]
            z = pl_module.encoder(x)
            x_reconstructed = pl_module.decoder(z)

            x = x.permute(0, 2, 3, 1).contiguous().cpu().data.numpy()
            ims_gt = Utility.convert_to_display(x)
            x_reconstructed = x_reconstructed.permute(0, 2, 3, 1).contiguous().cpu().data.numpy()

            ims_enc_dec = Utility.convert_to_display(x_reconstructed)

            # log to wandb
            self.wandb_logger.log_image("gt_vs_encDec", images=[ims_gt, ims_enc_dec])
            break
