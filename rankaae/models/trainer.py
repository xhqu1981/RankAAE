import shutil
import os
import logging
import itertools
import socket
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro, spearmanr

import torch
torch.autograd.set_detect_anomaly(True)
from torch import nn
from torch.optim.lr_scheduler import (
    ReduceLROnPlateau,
    CyclicLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    StepLR
)

from rankaae.models.model import (
    DiscriminatorFC,
    ExEncoder,
    ExDecoder
)
from rankaae.models.dataloader import get_dataloaders
from rankaae.utils.parameter import AE_CLS_DICT, OPTIM_DICT, Parameters
from rankaae.utils.functions import (
    kendall_constraint, 
    recon_loss, 
    mutual_info_loss, 
    smoothness_loss, 
    discriminator_loss,
    generator_loss,
    adversarial_loss,
    exscf_loss,
    alpha
)


class Trainer:
    
    metric_weights = [1.0, -1.0, -0.01, -1.0, -1.0]
    gau_kernel_size = 17

    def __init__(
        self, 
        encoder, decoder, discriminator, device, train_loader, val_loader,
        verbose=True, work_dir='.', tb_logdir="runs", 
        config_parameters = Parameters({}), # initialize Parameters with an empty dictonary.
        logger = logging.getLogger("training"),
        loss_logger = logging.getLogger("losses")
    ):
        self.logger = logger
        self.loss_logger = loss_logger # for recording losses as a function of epochs
        self.device = device
        self.encoder = encoder.to(self.device)
        self.decoder = decoder.to(self.device)
        self.discriminator = discriminator.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.verbose = verbose
        self.work_dir = work_dir
        self.tb_logdir = tb_logdir

        # update name space with config_parameters dictionary
        self.epoch_stop_smooth = 500 # default value in case it's not in fix_config, (to deprecate).
        self.__dict__.update(config_parameters.to_dict())
        self.load_optimizers()
        self.load_schedulers()


    def train(self, callback=None):
        if self.verbose:
            para_info = torch.__config__.parallel_info()
            self.logger.info(para_info)

        # loss functions
        mse_loss = nn.MSELoss().to(self.device)
        bce_lgt_loss = nn.BCEWithLogitsLoss().to(self.device)

        # train network
        best_combined_metric = -float('inf') # Initialize a guess for best combined metric.
        chkpt_dir = f"{self.work_dir}/checkpoints"
        if not os.path.exists(chkpt_dir):
            os.makedirs(chkpt_dir, exist_ok=True)
        best_chpt_file = None
        metrics = None
        
        # Record first line of loss values
        self.loss_logger.info( 
                "Epoch,Train_L1,Val_L1,Train_ExSCF,Val_ExSCF,Train_D,Val_D,Train_G,Val_G,Train_Aux,Val_Aux,Train_Recon,"
                "Val_Recon,Train_Smooth,Val_Smooth,Train_Mutual_Info,Val_Mutual_Info"
        )
        
        for epoch in range(self.max_epoch):
            self.encoder.train()
            self.decoder.train()
            self.discriminator.train()

            if self.gradient_reversal:
                alpha_ = alpha(epoch/self.max_epoch, self.alpha_flat_step, self.alpha_limit)

            # Loop through the labeled and unlabeled dataset getting one batch of samples from each
            # The batch size has to be a divisor of the size of the dataset or it will return
            # invalid samples
            for spec_in, aux_in in self.train_loader:
                spec_in = spec_in.to(self.device)
                n_aux = self.__dict__.get('n_aux', 0)
                if self.train_loader.dataset.aux is None:
                    aux_in = None
                else:
                    assert len(aux_in.size()) == 2
                    assert n_aux == aux_in.size()[-1]
                    aux_in = aux_in.to(self.device)
                
                spec_in_no_noise = spec_in.clone()
                spec_in += torch.randn_like(spec_in, requires_grad=False) * self.spec_noise
                if self.__dict__.get('randomize_spec_height', False):
                    spec_in *= 1.0 + torch.randn(spec_in.size()[0], device=spec_in.device, requires_grad=False)[:, None] * self.spec_height_noise
                styles = self.encoder(spec_in) # exclude the free style
                spec_out = self.decoder(styles) # reconstructed spectra

                # Use gradient reversal method or standard GAN structure
                if self.gradient_reversal and self.optimizers["adversarial"] is not None:
                    self.zerograd()
                    dis_loss_train = adversarial_loss(
                        spec_in, styles, self.discriminator, alpha_,
                        batch_size=self.z_sample_batch_size, 
                        nll_loss=bce_lgt_loss, 
                        device=self.device
                    )
                    dis_loss_train.backward()
                    self.optimizers["adversarial"].step()
                    gen_loss_train = torch.tensor(0.0)
                elif self.optimizers["discriminator"] is not None and self.optimizers["generator"] is not None:
                    # Init gradients, discriminator loss
                    self.zerograd()
                    styles = self.encoder(spec_in)

                    dis_loss_train = discriminator_loss(
                        styles, self.discriminator, 
                        batch_size=self.z_sample_batch_size, 
                        loss_fn=bce_lgt_loss,
                        device=self.device
                    )
                    dis_loss_train.backward()
                    self.optimizers["discriminator"].step()

                    # Init gradients, generator loss
                    self.zerograd()
                    gen_loss_train = generator_loss(
                        spec_in, self.encoder, self.discriminator, 
                        loss_fn=bce_lgt_loss,
                        device=self.device
                    )
                    gen_loss_train.backward()
                    self.optimizers["generator"].step()
                else:
                    dis_loss_train = torch.tensor(0.0)
                    gen_loss_train = torch.tensor(0.0)
                
                if self.optimizers["correlation"] is not None:
                    # Kendall constraint
                    self.zerograd()
                    styles = self.encoder(spec_in)
                    aux_loss_train = kendall_constraint(
                        aux_in, styles[:,:n_aux], 
                        force_balance=self.kendall_force_balance,
                        device=self.device)
                    aux_regen_ratio = self.__dict__.get('aux_regen_ratio', 0.0)
                    if aux_regen_ratio > 0.0:
                        styles = self.encoder(self.decoder(styles))
                        aux_loss_2 = kendall_constraint(
                            aux_in, styles[:,:n_aux], 
                            force_balance=self.kendall_force_balance,
                            device=self.device)
                        aux_loss_train = (1.0 - aux_regen_ratio) * aux_loss_train + aux_regen_ratio * aux_loss_2
                    aux_loss_train.backward()
                    self.optimizers["correlation"].step()
                else:
                    aux_loss_train = torch.tensor(0.0, dtype=torch.float32)

                # Init gradients, reconstruction loss
                self.zerograd()
                spec_out  = self.decoder(self.encoder(spec_in))
                recon_loss_train = recon_loss(
                    spec_in_no_noise, spec_out, 
                    scale=self.use_flex_spec_target,
                    device=self.device
                )
                recon_loss_train.backward()
                self.optimizers["reconstruction"].step()

                # Init gradients, mutual information loss
                if self.optimizers["mutual_info"] is not None:
                    self.zerograd()
                    styles = self.encoder(spec_in)
                    mutual_info_loss_train = mutual_info_loss(
                        self.z_sample_batch_size, self.nstyle,
                        encoder=self.encoder, 
                        decoder=self.decoder,
                        n_aux=n_aux, 
                        mse_loss=mse_loss, 
                        device=self.device
                    )
                    mutual_info_loss_train.backward()
                    self.optimizers["mutual_info"].step()
                else:
                    mutual_info_loss_train = torch.tensor(0.0)

                # Init gradients, smoothness loss
                if epoch < self.epoch_stop_smooth and self.optimizers["smoothness"] is not None: 
                    # turn off smooth loss after 500
                    self.zerograd()
                    smooth_loss_train = smoothness_loss(
                        self.z_sample_batch_size, self.nstyle, self.decoder, 
                        gs_kernel_size=self.gau_kernel_size,
                        device=self.device,
                        layered_smooth=self.__dict__.get('layered_smooth', False),
                        encoder=self.encoder
                    )
                    self.optimizers["smoothness"].step()
                else:
                    smooth_loss_train = torch.tensor(0.0)

                # L1 regularization to encourage sparseness
                if self.optimizers["l1_regularization"] is not None:
                    self.zerograd()
                    n_params = sum(
                        [torch.numel(p) for p in 
                         itertools.chain(self.encoder.get_training_parameters(), 
                                         self.decoder.get_training_parameters())])
                    l1_loss = torch.sum(torch.stack(
                        [torch.sum(torch.abs(p)) for p in 
                         itertools.chain(self.encoder.get_training_parameters(), 
                                         self.decoder.get_training_parameters())])) / n_params
                    l1_loss.backward()
                    self.optimizers["l1_regularization"].step()
                else:
                    l1_loss = torch.tensor(0.0)

                if self.optimizers["exscf"] is not None:
                    self.zerograd()
                    exscf_loss_train = exscf_loss(
                        self.z_sample_batch_size, self.nstyle, self.encoder, self.decoder,
                        mse_loss=mse_loss, device=self.device)
                    exscf_loss_train.backward()
                    self.optimizers["exscf"].step()
                else:
                    exscf_loss_train = torch.tensor(0.0)
                  
                # Init gradients
                self.zerograd()

            ### Validation ###
            self.encoder.eval()
            self.decoder.eval()
            self.discriminator.eval()
            
            spec_in_val, aux_in_val = [torch.cat(x, dim=0) for x in zip(*list(self.val_loader))]
            spec_in_val = spec_in_val.to(self.device)
            z = self.encoder(spec_in_val)
            spec_out_val = self.decoder(z)

            if self.train_loader.dataset.aux is None:
                aux_in = None
            else:
                assert len(aux_in_val.size()) == 2
                assert n_aux == aux_in_val.size()[-1]
                aux_in_val = aux_in_val.to(self.device)
            
            recon_loss_val = recon_loss(
                spec_in_val, 
                spec_out_val, 
                mse_loss=mse_loss, 
                device=self.device
            )

            if n_aux > 0:
                aux_loss_val = kendall_constraint(
                    aux_in_val, 
                    z[:,:n_aux], 
                    force_balance=False,
                    device=self.device,
                    validation_only=True
                )
            else:
                aux_loss_val = torch.tensor(0.0)

            smooth_loss_val = smoothness_loss(
                self.z_sample_batch_size, self.nstyle, self.decoder, 
                gs_kernel_size=self.gau_kernel_size,
                device=self.device
            )

            mutual_info_loss_val =  mutual_info_loss(
                self.z_sample_batch_size, self.nstyle,
                encoder=self.encoder, 
                decoder=self.decoder, 
                n_aux=n_aux,
                mse_loss=mse_loss, 
                device=self.device
            )

            if self.gradient_reversal and self.optimizers["adversarial"] is not None:
                dis_loss_val = adversarial_loss(
                    spec_in_val, z, self.discriminator, alpha_,
                    batch_size=self.z_sample_batch_size, 
                    nll_loss=bce_lgt_loss, 
                    device=self.device
                )
                gen_loss_val = torch.tensor(0.0)
            elif self.optimizers["discriminator"] is not None and self.optimizers["generator"] is not None:
                dis_loss_val = discriminator_loss(
                    z, self.discriminator, 
                    batch_size=self.z_sample_batch_size,
                    loss_fn=bce_lgt_loss,
                    device=self.device
                )
                gen_loss_val = generator_loss(
                    spec_in_val, 
                    self.encoder, 
                    self.discriminator, 
                    loss_fn=bce_lgt_loss, 
                    device=self.device
                )
            else:
                dis_loss_val = torch.tensor(0.0)
                gen_loss_val = torch.tensor(0.0)

            if self.optimizers["exscf"] is not None:
                exscf_loss_val = exscf_loss(
                    self.z_sample_batch_size, self.nstyle, self.encoder, self.decoder,
                    mse_loss=mse_loss, device=self.device)
            else:
                exscf_loss_val = torch.tensor(0.0)
                
            # Write losses to a file
            if epoch % 10 == 0:
                self.loss_logger.info(
                    f"{epoch:d},\t"
                    f"{l1_loss.item():.6f},\t{l1_loss.item():.6f},\t"
                    f"{exscf_loss_train.item():.6f},\t{exscf_loss_val.item():.6f},\t"
                    f"{dis_loss_train.item():.6f},\t{dis_loss_val.item():.6f},\t"
                    f"{gen_loss_train.item():.6f},\t{gen_loss_val.item():.6f},\t"
                    f"{aux_loss_train.item():.6f},\t{aux_loss_val.item():.6f},\t"
                    f"{recon_loss_train.item():.6f},\t{recon_loss_val.item():.6f},\t"
                    f"{smooth_loss_train.item():.6f},\t{smooth_loss_val.item():.6f},\t"
                    f"{mutual_info_loss_train.item():.6f},\t{mutual_info_loss_val.item():.6f}"
                )
            
            model_dict = {"Encoder": self.encoder,
                          "Decoder": self.decoder,
                          "Style Discriminator": self.discriminator}
            
            style_np = z.detach().clone().cpu().numpy().T
            style_shapiro = [shapiro(x).statistic for x in style_np]
            style_coupling = np.max(np.fabs(
                [
                    spearmanr(style_np[j1], style_np[j2]).correlation
                    for j1, j2 in itertools.combinations(range(style_np.shape[0]), 2)
                ]
            ))
            metrics = [min(style_shapiro), recon_loss_val.item(), mutual_info_loss_val.item(), style_coupling,
                       aux_loss_val.item() if n_aux > 0 else 0]
            
            combined_metric = (np.array(self.metric_weights) * np.array(metrics)).sum()
            if isinstance(self.encoder, ExEncoder):
                combined_metric = -aux_loss_val

            if combined_metric > best_combined_metric:
                best_combined_metric = combined_metric
                best_chpt_file = f"{chkpt_dir}/epoch_{epoch:06d}_loss_{combined_metric:07.6g}.pt"
                torch.save(model_dict, best_chpt_file)

            for _, sch in self.schedulers.items():
                if isinstance(sch, ReduceLROnPlateau):
                    sch.step(combined_metric)
                else:
                    sch.step()

            if callback is not None:
                callback(epoch, metrics)
            
        # save the final model
        torch.save(model_dict, f'{self.work_dir}/final.pt')

        if best_chpt_file is not None:
            shutil.copy2(best_chpt_file, f'{self.work_dir}/best.pt')

        return metrics


    def zerograd(self):
        self.encoder.zero_grad()
        self.decoder.zero_grad()
        self.discriminator.zero_grad()

    def get_style_distribution_plot(self, z):
        # noinspection PyTypeChecker
        fig, ax_list = plt.subplots(
            self.nstyle, 1, sharex=True, sharey=True, figsize=(9, 12))
        for istyle, ax in zip(range(self.nstyle), ax_list):
            sns.histplot(z[:, istyle], kde=False, color='blue', bins=np.arange(-3.0, 3.01, 0.2),
                         ax=ax, element="step")
        return fig


    def load_optimizers(self):
        opt_cls = OPTIM_DICT[self.optimizer_name]

        recon_optimizer = opt_cls(
            params = [
                {'params': self.decoder.get_training_parameters()}
            ] if self.__dict__.get('reconn_decoder_only', False) else [
                {'params': self.encoder.get_training_parameters()},
                {'params': self.decoder.get_training_parameters()}                  
            ],
            lr = self.lr_ratio_Reconn * self.lr_base,
            weight_decay = self.weight_decay)

        if self.__dict__.get('lr_ratio_Mutual', -1) > 0:
            mutual_info_optimizer = opt_cls(
                params = [{'params': self.encoder.get_training_parameters()}, 
                          {'params': self.decoder.get_training_parameters()}],
                lr = self.lr_ratio_Mutual * self.lr_base,
                weight_decay = self.weight_decay)
        else:
            mutual_info_optimizer = None

        if self.__dict__.get('lr_ratio_Smooth', -1) > 0:
            smooth_optimizer = opt_cls(
                params = [{'params': self.encoder.get_training_parameters()}, 
                          {'params': self.decoder.get_training_parameters()}] \
                          if isinstance(self.encoder, ExEncoder) \
                          else [{'params': self.decoder.get_training_parameters()}],
                lr = self.lr_ratio_Smooth * self.lr_base,
                weight_decay = self.weight_decay)
        else:
            smooth_optimizer = None

        if self.__dict__.get('lr_ratio_Corr', -1) > 0:
            corr_optimizer = opt_cls(
                [{'params': self.encoder.get_training_parameters()}],
                lr = self.lr_ratio_Corr * self.lr_base,
                weight_decay = self.weight_decay)
        else:
            corr_optimizer = None

        if self.__dict__.get('lr_ratio_dis', -1) > 0:
            dis_optimizer = opt_cls(
                [{'params': self.discriminator.parameters()}],
                lr = self.lr_ratio_dis * self.lr_base,
                weight_decay = self.weight_decay,
                betas = (self.dis_beta * 0.9, self.dis_beta * 0.009 + 0.99))
        else:
            dis_optimizer = None

        if self.__dict__.get('lr_ratio_gen', -1) > 0:
            gen_optimizer = opt_cls(
                [{'params': self.encoder.get_training_parameters()}],
                lr = self.lr_ratio_gen * self.lr_base,
                weight_decay = self.weight_decay,
                betas = (self.gen_beta * 0.9, self.gen_beta * 0.009 + 0.99))
        else:
            gen_optimizer = None

        if self.__dict__.get('lr_ratio_adv', -1) > 0:
            adv_optimizer = opt_cls(
                [{'params': self.discriminator.parameters()},
                 {'params': self.encoder.get_training_parameters()}],
                lr = self.lr_ratio_adv * self.lr_base,
                weight_decay = self.weight_decay,
                betas = (self.dis_beta * 0.9, self.dis_beta * 0.009 + 0.99))
        else:
            adv_optimizer = None

        if self.__dict__.get('lr_ratio_L1', -1) > 0:
            l1_regularization = opt_cls(
                params = [{'params': self.encoder.get_training_parameters()}, 
                          {'params': self.decoder.get_training_parameters()}],
                lr = self.lr_ratio_L1 * self.lr_base,
                weight_decay = 0)
        else:
            l1_regularization = None

        if self.__dict__.get('lr_ratio_exscf', -1) > 0:
            assert isinstance(self.encoder, ExEncoder)
            assert isinstance(self.decoder, ExDecoder)
            exscf_optimizer = opt_cls(
                params = [
                    {'params': self.encoder.get_training_parameters()}
                ] if self.__dict__.get('exscf_encoder_only', False) else [
                    {'params': self.encoder.get_training_parameters()},
                    {'params': self.decoder.get_training_parameters()}                  
                ],
                lr = self.lr_ratio_exscf * self.lr_base,
                weight_decay = self.weight_decay)
        else:
            exscf_optimizer = None

        self.optimizers = {
            "reconstruction": recon_optimizer,
            "mutual_info": mutual_info_optimizer,
            "smoothness": smooth_optimizer,
            "correlation": corr_optimizer,
            "discriminator": dis_optimizer,
            "generator": gen_optimizer,
            "adversarial": adv_optimizer,
            "l1_regularization": l1_regularization,
            "exscf": exscf_optimizer
        }


    def load_schedulers(self):
        def create_scheduler(optimizer):
            sch_name = self.__dict__.get('scheduler', 'ReduceLROnPlateau')
            if sch_name == 'ReduceLROnPlateau':
                return ReduceLROnPlateau(
                    optimizer, mode="max", factor=self.sch_factor, patience=self.sch_patience, 
                    cooldown=0, threshold=0.01,verbose=self.verbose)
            elif sch_name == 'CyclicLR':
                return CyclicLR(optimizer, base_lr=optimizer.param_groups[0]['lr']*1.0E-4, 
                                max_lr=optimizer.param_groups[0]['lr'], cycle_momentum=False,
                                step_size_up=self.sch_patience, step_size_down=self.sch_patience)
            elif sch_name == 'CosineAnnealingLR':
                return CosineAnnealingLR(optimizer, T_max=self.sch_patience, eta_min=1.0E-8)
            elif sch_name == 'CosineAnnealingWarmRestarts':
                return CosineAnnealingWarmRestarts(optimizer, T_0=self.sch_patience, eta_min=1.0E-8)
            elif sch_name == 'StepLR':
                return StepLR(optimizer, step_size=self.sch_patience, gamma=self.sch_factor)
            else:
                raise ValueError(f"Schedule {sch_name} is not recognized")
                
        self.schedulers = {name:create_scheduler(optimizer)
            for name, optimizer in self.optimizers.items() if optimizer is not None}


    @classmethod
    def from_data(
        cls, csv_fn, 
        igpu=0, verbose=True, work_dir='.', 
        train_ratio=0.7, validation_ratio=0.15, test_ratio=0.15, 
        config_parameters = Parameters({}),
        logger = logging.getLogger("from_data"),
        loss_logger = logging.getLogger("losses")
    ):

        p = config_parameters
        assert p.ae_form in AE_CLS_DICT

        # load training and validation dataset
        dl_train, dl_val, _ = get_dataloaders(
            csv_fn, p.batch_size, (train_ratio, validation_ratio, test_ratio), n_aux=p.n_aux)

        # Use GPU if possible
        if torch.cuda.is_available():
            if verbose:
                logger.info("Use GPU")
                logger.info(f"Running on {socket.gethostname()} with GPU #{igpu + 1}\n")
            device = torch.device(f"cuda:{igpu}")
            for loader in [dl_train, dl_val]:
                loader.pin_memory = False
        else:
            if verbose:
                logger.warn("Use Slow CPU!")
            device = torch.device("cpu")

        if 'initial_guess_dir' in p.__dict__:
            assert os.path.isdir(p.initial_guess_dir)
            # load encoder, decoder and discriminator from file
            prev_fn = os.path.join(p.initial_guess_dir, *work_dir.split('/')[-2:], 'final.pt')
            logger.info(f"Reading model initial guess from {prev_fn}")
            mt = torch.load(prev_fn, map_location=device)
            two_hot_generator = torch.load(p.twohot_fn, map_location=device)
            encoder = ExEncoder(p.dim_in, enclosing_encoder=mt['Encoder'],
                                two_hot_generator=two_hot_generator,
                                gate_window=p.get('gate_window', 13),
                                n_gate_encoder_layers=p.get('n_gate_encoder_layers', 3),
                                n_gate_decoder_layers=p.get('n_gate_decoder_layers', 3),
                                gate_hidden_size=p.get('gate_hidden_size', 64),
                                gate_latent_dim=p.get('gate_latent_dim', 1),
                                activation=p.get('ex_activation', 'Swish'),
                                n_polynomial_order=p.get('n_polynomial_order', 3),
                                n_polynomial_points=p.get('n_polynomial_points', 10),
                                padding_mode=p.get('padding_mode', 'stretch'))
            decoder = ExDecoder(p.dim_out, enclosing_decoder=mt['Decoder'],
                                two_hot_generator=two_hot_generator,
                                gate_window=p.get('gate_window', 13),
                                n_gate_encoder_layers=p.get('n_gate_encoder_layers', 3),
                                n_gate_decoder_layers=p.get('n_gate_decoder_layers', 3),
                                gate_hidden_size=p.get('gate_hidden_size', 64),
                                gate_latent_dim=p.get('gate_latent_dim', 1),
                                activation=p.get('ex_activation', 'Swish'),
                                n_polynomial_order=p.get('n_polynomial_order', 3),
                                n_polynomial_points=p.get('n_polynomial_points', 10),
                                padding_mode=p.get('padding_mode', 'stretch'))
            discriminator = mt['Style Discriminator']
        else:
            # Generate encoder, decoder and discriminator
            logger.info("Generate model initial guess using random numbers")
            encoder = AE_CLS_DICT[p.ae_form]["encoder"](
                nstyle = p.nstyle, 
                dim_in = p.dim_in, 
                n_layers = p.n_layers,
                activation=p.get('activation', "Swish"),
            )
            decoder = AE_CLS_DICT[p.ae_form]["decoder"](
                nstyle = p.nstyle, 
                activation=p.get('activation', "Swish"),
                last_layer_activation = p.decoder_activation, 
                dim_out = p.dim_out,
                n_layers = p.n_layers
            )
            discriminator = DiscriminatorFC(
                nstyle=p.nstyle, noise=p.dis_noise,
                layers = p.FC_discriminator_layers,
                activation=p.get('activation', "Swish"),
            )

        for net in [encoder, decoder, discriminator]:
            logger.info(repr(net))
            net.to(device)

        # Load trainer
        trainer = Trainer(
            encoder, decoder, discriminator, device, dl_train, dl_val,
            verbose=verbose, work_dir=work_dir,
            config_parameters=p, logger=logger, loss_logger=loss_logger
        )
        return trainer


 