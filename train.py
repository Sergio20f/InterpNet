import torch
import torch.nn as nn
import numpy as np
import itertools

from attn_autoencoder import AE_attn
from loss import GammaContrastReconLoss
from utils import printProgressBar


class AE_module(nn.Module):
    def __init__(self, config):
        super(AE_module, self).__init__()

        self.verbosity = config.get("verbosity", 0)

        if self.verbosity != 0:
            print("Initialise the model")

        self.device = config.get("device", torch.device("cpu"))
        if type(self.device) is not torch.device:
            self.device = torch.device(self.device)
        if self.verbosity != 0:
            print(f"device: {self.device}")

        seed = 42
        torch.manual_seed(seed=seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.model = AE_attn(config)
        # Could add projection head

        self.criterion = GammaContrastReconLoss(
            gamma=config.get("training::gamma", 0.5),
            reduction="mean",
            batch_size=config.get("trainset::batchsize", 64),
            temperature=config.get("training::temperature", 0.1),
            device=self.device,
            config=config,            
        )
        
        # send model and criterion to device
        self.model.to(self.device)
        self.criterion.to(self.device)

        # set optimizer
        self.set_optimizer(config)
        # set trackers
        self.best_epoch = None
        self.loss_best = None
        self.best_checkpoint = None

        # mean loss for r^2
        self.loss_mean = None

        # init scheduler
        self.set_scheduler(config)

    def forward(self, x):
        z, y = self.model.forward(x)
        return z, y
    
    def set_optimizer(self, config):
        params_lst = [self.model.parameters(), self.criterion.parameters()]
        # Combine the two sets of parameters into a single iterator
        params = itertools.chain(*params_lst) # * to unpack params_lst into separate args
        if config.get("optim::optimizer", "adam") == "sgd":
            self.optimizer = torch.optim.SGD(
                params,
                lr=config.get("optim::lr", 3e-4),
                momentum=config.get("optim::momentum", 0.9),
                weight_decay=config.get("optim::wd", 3e-5),
            )
        if config.get("optim::optimizer", "adam") == "adam":
            self.optimizer = torch.optim.Adam(
                params,
                lr=config.get("optim::lr", 3e-4),
                weight_decay=config.get("optim::wd", 3e-5),
            )
    
    def set_scheduler(self, config):
        # For now keep it simple, complete function in def_simclr_ae_module.py
        self.scheduler = None

    def save_model(self, epoch, performance_dict, path=None):
        if path is not None:
            fname = path.joinpath(f"model_epoch_{epoch}.pt")
            # save model state-dict
            performance_dict["state_dict"] = self.model.state_dict()
            # save optimizer state-dict
            performance_dict["optimizer_state"] = self.optimizer.state_dict()
            torch.save(performance_dict, fname)
        return None
    
    def train_step(self, x_i, x_j):
        # zero grads before training steps
        self.optimizer.zero_grad()
        # forward pass with both views
        # Views: different reps of the input data (common in CL)
        z_i, y_i = self.forward(x_i)
        z_j, y_j = self.forward(x_j)
        # cat y_i, y_j and x_i, x_j
        x = torch.cat([x_i, x_j], dim=0)
        y = torch.cat([y_i, y_j], dim=0)
        # compute loss
        loss, loss_contrast, loss_recon = self.criterion(z_i=z_i, z_j=z_j, y=y, t=x)
        # prop loss backwards to
        loss.backward()
        # update parameters
        self.optimizer.step()
        return loss.item(), loss_contrast.item(), loss_recon.item()
    
    def train(self, trainloader, epoch, writer=None, tf_out=10):
        # trainloader is originally defined in FastTensorDataLoader.py
        if self.verbosity != 0:
            print(f"train epoch {epoch}")
        # set model to trainig mode
        self.model.train()

        if self.verbosity != 0:
            printProgressBar(
                0,
                len(trainloader),
                prefix="Batch Progress:",
                suffix="Complete",
                length=50,
            )
        
        # setup cumulative loss and accuracy
        loss_acc = 0
        loss_acc_contr = 0
        loss_acc_recon = 0
        n_data = 0
        # enter loop over batches
        for idx, data in enumerate(trainloader):
            x_i, l_i, x_j, _ = data
            # send to device
            x_i = x_i.to(self.device)
            x_j = x_j.to(self.device)  # take one training step

            if self.verbosity != 0:
                printProgressBar(
                    idx + 1,
                    len(trainloader),
                    prefix="Batch Progress:",
                    suffix="Complete",
                    length=50,
                )
            
            # Compute loss
            loss, loss_contr, loss_recon = self.train_step(x_i, x_j)
            # scale loss with batchsize (get's normalized later)
            loss_acc += loss * len(l_i)
            loss_acc_contr += loss_contr * len(l_i)
            loss_acc_recon += loss_recon * len(l_i)
            n_data += len(l_i)
            # logging
            if idx > 0 and idx % tf_out == 0:
                loss_running = loss_acc / n_data
                loss_running_contr = loss_acc_contr / n_data
                loss_running_recon = loss_acc_recon / n_data
                if self.verbosity > 0:
                    print(
                        f"epoch {epoch} - batch {idx}/{len(trainloader)} ::: loss: {loss_running}; loss_contrast: {loss_running_contr}, loss_reconstruction: {loss_running_recon}"
                    )

        # ...