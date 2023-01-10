from models.SimCLRModel import SimCLRModel
import pytorch_lightning as pl
from datasets.SimCLR_cvrt import SimCLRDataModule
import yaml
import argparse
import os
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import torch.utils.data as data
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help='Config file containing the summarization arguments')
args = parser.parse_args()

stream = open(args.config, 'r')
config = yaml.load(stream, Loader=yaml.FullLoader)
stream.close()

BATCH_SIZE=config["batch_size"]
NUM_WORKERS=config["num_workers"]
CHECKPOINT_PATH = config["checkpoint_path"]

simclr_data = SimCLRDataModule(config["data_dir"], num_workers=NUM_WORKERS, batch_size=BATCH_SIZE)

def train_simclr(batch_size=64, max_epochs=5, num_workers=1, **kwargs):

    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH),
                         accelerator="gpu",
                         devices=1,
                         max_epochs=max_epochs,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode='max', monitor='val_acc_top5'),
                                    LearningRateMonitor('epoch')])

    trainer.logger._default_hp_metric = None # Optional logging argument that we don't need

    train_loader = data.DataLoader(simclr_data.train_set, batch_size=batch_size, shuffle=True, 
                                    drop_last=True, num_workers=num_workers)
    val_loader = data.DataLoader(simclr_data.val_set, batch_size=batch_size, shuffle=False, 
                                    drop_last=False, num_workers=num_workers)

    pl.seed_everything(config["seed"]) # To be reproducable
    model = SimCLRModel(max_epochs=max_epochs, **kwargs)
    trainer.fit(model, train_loader, val_loader)
    model = SimCLRModel.load_from_checkpoint(trainer.checkpoint_callback.best_model_path) # Load best checkpoint after training

    return model


simclr_model = train_simclr(batch_size=config["batch_size"], 
                            hidden_dim=config["mlp_hidden_dim"],
                            out_dim=config["mlp_dim"], 
                            lr=config["lr"], 
                            temperature=0.07,
                            max_epochs=config["max_epochs"])

torch.save(simclr_model.convnet.state_dict(), os.path.join(config["checkpoint_path"], "backbone.pt"))
torch.save(simclr_model.mlp.state_dict(), os.path.join(config["checkpoint_path"], "mlp.pt"))