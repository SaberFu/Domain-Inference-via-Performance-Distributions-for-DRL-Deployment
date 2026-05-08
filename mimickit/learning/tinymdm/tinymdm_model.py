import torch
import torch.nn.functional as F
import torch.nn as nn
import yaml

from diffusers.schedulers import DDPMScheduler, DDIMScheduler

from learning.tinymdm.base_model import KinematicBaseModel
from learning.tinymdm.arch import TinyStableMotionDiTModel, CondTinyStableMotionDiTModel
from learning.tinymdm.EMA import EMA
from learning.tinymdm.cfg_model import ClassifierFreeSampleModel
from learning.normalizer import Normalizer

class TinyMDMModel(KinematicBaseModel):
    def __init__(self, config):
        super().__init__(config)

        with open(config["env_config"], "r") as stream:
            env_config = yaml.safe_load(stream)
        
        self.num_disc_obs_steps = env_config["num_disc_obs_steps"]
        self._obs_dtype = torch.float32

        self.T = config["T"]
        self._device = config["device"]
        self.input_dim = config["input_dim"]
        self.input_channel = int(self.input_dim / self.num_disc_obs_steps)
        self.num_layers = config["num_layers"]
        self.dropout = config.get("dropout", 0.)
        self.estimate_mode = config["estimate_mode"]
        self.loss_type = config["loss_type"]
        self.schedule_mode = config["noise_schedule_mode"]
        self.arch_name = config["arch_name"]
        self.model_ema = config.get("model_ema", False)
        self.config = config

        if self.arch_name == "DiT":
            print("Using DiT architecture")
            self.num_attention_heads = config.get("num_attention_heads", 4)
            self.attention_head_dim = config.get("attention_head_dim", 64)
            self.dmodel = TinyStableMotionDiTModel(
                in_channels=self.input_channel,
                num_layers=self.num_layers,
                attention_head_dim=self.attention_head_dim,
                num_attention_heads=self.num_attention_heads,
                out_channels=self.input_channel,
                dropout=self.dropout,
            )
        elif self.arch_name == "CondDiT":
            print("Using CondDiT architecture")
            self.num_attention_heads = config.get("num_attention_heads", 4)
            self.attention_head_dim = config.get("attention_head_dim", 64)
            self.dmodel = CondTinyStableMotionDiTModel(
                in_channels=self.input_channel,
                num_layers=self.num_layers,
                attention_head_dim=self.attention_head_dim,
                num_attention_heads=self.num_attention_heads,
                out_channels=self.input_channel,
                dropout=self.dropout,
                num_class=config.get("num_class", 0),
                cfg_dropout=config.get("cfg_dropout", 0.),
            )
        else:
            raise ValueError(f"Unknown architecture: {self.arch_name}")
        
        if self.model_ema:
            print("Using model EMA")
            self.ema_dmodel = EMA(self.dmodel,
                                 beta=config["model_ema_decay"],
                                 update_every=config["model_ema_steps"],
                                 update_after_step=config["model_ema_update_after"]
                                 )
      
        self.diffusion_scheduler = DDPMScheduler(
            num_train_timesteps=self.T,
            beta_schedule=self.schedule_mode,
            prediction_type=self.estimate_mode,
            clip_sample=False,
        )
        self.ddim_scheduler = DDIMScheduler(
            num_train_timesteps=self.T,
            beta_schedule=self.schedule_mode,
            prediction_type=self.estimate_mode,
            clip_sample=False,
        )
        self.ddim_scheduler.set_timesteps(self.T, device=self._device)


        self.diffusion_scheduler.alphas_cumprod = self.diffusion_scheduler.alphas_cumprod.to(self._device)
        self.diffusion_scheduler.timesteps = self.diffusion_scheduler.timesteps.to(self._device)

        self._init_normalizer()
        return
    
    def _init_normalizer(self):
        self.obs_normalizer = Normalizer(
            (self.input_channel),
            device=self._device,
            dtype=self._obs_dtype,
            std_clip=self.config["normalizer_std_clip"],
        )
        return
    
    def update_normalizer(self, samples):
        self.obs_normalizer.record(samples.reshape(-1, self.input_channel))
        self.obs_normalizer.update()
        return
    
    def unnormalize(self, norm_samples):
        return self.obs_normalizer.unnormalize(norm_samples.reshape(-1, self.input_channel)).reshape(norm_samples.shape)
    
    def normalize(self, samples):
        return self.obs_normalizer.normalize(samples.reshape(-1, self.input_channel)).reshape(samples.shape)

    def _get_epsilon(self, t, x_noised, pred_x_start):
        alpha_prod_t = self.diffusion_scheduler.alphas_cumprod[t].unsqueeze(-1)
        beta_prod_t = 1 - alpha_prod_t
        pred_epsilon = (x_noised - pred_x_start * alpha_prod_t ** (0.5)) / beta_prod_t ** (0.5)
        return pred_epsilon

    def _get_sampling_denoiser(self, use_ema=False, cfg_scale=1.0):
        if use_ema:
            denoiser = self.ema_dmodel
        else:
            denoiser = self.dmodel

        if cfg_scale != 1.0:
            denoiser = ClassifierFreeSampleModel(denoiser)

        return denoiser

    @torch.no_grad()
    def _sample_with_scheduler(self, denoiser, scheduler, timesteps, shape, batch_size, device, **kwargs):
        x_t = torch.randn(batch_size, *shape, device=device)

        for t in timesteps:
            timestep = torch.full((batch_size,), int(t), dtype=torch.long, device=device)
            pred = denoiser(x_t, timestep, **kwargs)
            scheduler_output = scheduler.step(pred, t, x_t)
            x_t = scheduler_output.prev_sample

        x_0 = scheduler_output.pred_original_sample
        return x_0

    def forward(self, x, timesteps=None, **kwargs):
        bsz = x.shape[0]
        device = x.device
        if timesteps is None:
            timesteps = torch.randint(0, len(self.diffusion_scheduler.timesteps), (bsz,), device=device)
        noise = torch.randn_like(x)

        noised_x = self.diffusion_scheduler.add_noise(
            original_samples=x,
            noise=noise,
            timesteps=timesteps
        )

        input = noised_x
        pred = self.dmodel(input, timesteps, **kwargs)
    
        if self.diffusion_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.diffusion_scheduler.config.prediction_type == "sample":
            target = x
        elif self.diffusion_scheduler.config.prediction_type == "v_prediction":
            target = self.diffusion_scheduler.get_velocity(x, noise, timesteps)
        else:
            raise NotImplementedError
            
        if self.loss_type == "l1":
            loss = torch.nn.functional.l1_loss(pred, target.squeeze())
        elif self.loss_type == "l2":
            loss = torch.nn.functional.mse_loss(pred, target.squeeze())
        else:
            raise NotImplementedError 
        
        return loss
    
    @torch.no_grad()
    def sample(self, shape, batch_size, device, sampler="ddpm", num_inference_steps=None, **kwargs):
        denoiser = self._get_sampling_denoiser(use_ema=False, cfg_scale=kwargs.get("cfg_scale", 1.0))

        if sampler == "ddpm":
            scheduler = self.diffusion_scheduler
            timesteps = scheduler.timesteps
        elif sampler == "ddim":
            if num_inference_steps is None:
                num_inference_steps = self.T
            self.ddim_scheduler.set_timesteps(num_inference_steps, device=device)
            scheduler = self.ddim_scheduler
            timesteps = scheduler.timesteps
        else:
            raise ValueError(f"Unsupported sampler: {sampler}")

        return self._sample_with_scheduler(denoiser, scheduler, timesteps, shape, batch_size, device, **kwargs)

    @torch.no_grad()
    def sample_dump(self, shape, batch_size, device, **kwargs):
        if kwargs.get("cfg_scale", 1.) != 1:
            denoiser = ClassifierFreeSampleModel(self.dmodel)
        else:
            denoiser = self.dmodel
        x_t = torch.randn(batch_size, *shape, device=device)
        intermid_sample = []
        for t in self.diffusion_scheduler.timesteps:
            timestep = torch.repeat_interleave(t, batch_size).to(device)
            pred = denoiser(x_t, timestep, **kwargs)
            scheduler_output = self.diffusion_scheduler.step(pred, t, x_t)
            x_t = scheduler_output.prev_sample
            intermid_sample.append(scheduler_output.pred_original_sample)
        return intermid_sample
    
    
    @torch.no_grad()
    def sample_ema(self, shape, batch_size, device, sampler="ddpm", num_inference_steps=None, **kwargs):
        denoiser = self._get_sampling_denoiser(use_ema=True, cfg_scale=kwargs.get("cfg_scale", 1.0))

        if sampler == "ddpm":
            scheduler = self.diffusion_scheduler
            timesteps = scheduler.timesteps
        elif sampler == "ddim":
            if num_inference_steps is None:
                num_inference_steps = self.T
            self.ddim_scheduler.set_timesteps(num_inference_steps, device=device)
            scheduler = self.ddim_scheduler
            timesteps = scheduler.timesteps
        else:
            raise ValueError(f"Unsupported sampler: {sampler}")

        return self._sample_with_scheduler(denoiser, scheduler, timesteps, shape, batch_size, device, **kwargs)
    
    @torch.no_grad()
    def ESM_SDS_loss(
        self,
        norm_x_obs,
        t_lst = None,
        **kwargs,
    ) -> torch.Tensor:
        denoiser = self._get_sampling_denoiser(
            use_ema=self.model_ema,
            cfg_scale=kwargs.get("cfg_scale", 1.0),
        )

        sds_losses = []
        bsz = norm_x_obs.shape[0]
        device = norm_x_obs.device

        for t_value in t_lst:
            t = torch.full((bsz,), t_value, dtype=torch.long, device=device)
            # add noise
            current_noise = torch.randn_like(norm_x_obs)
            noised_x_obs = self.diffusion_scheduler.add_noise(norm_x_obs, current_noise, t)
            _x_t = noised_x_obs.clone()

            # denoise
            temp_t = torch.full((bsz,), t_value, dtype=torch.long, device=device)
            _pred = denoiser(
                    _x_t,
                    timestep=temp_t,
                    **kwargs,
                )
            re = self.ddim_scheduler.step(_pred, t_value, _x_t)
            _x_t, x_0_pred = re.prev_sample, re.pred_original_sample
                
            eps_pred = self._get_epsilon(t=t, x_noised=noised_x_obs, pred_x_start=x_0_pred)
            pred_err = eps_pred - current_noise

            mean_squared_error = torch.mean((pred_err) ** 2, dim=list(range(1, noised_x_obs.dim())))
            sds_losses.append(mean_squared_error)

        sds_losses = torch.stack(sds_losses, dim=1)
        return sds_losses