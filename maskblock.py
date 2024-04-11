
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from math import sqrt


Linear = nn.Linear
ConvTranspose2d = nn.ConvTranspose2d


def Conv1d(*args, **kwargs):
  layer = nn.Conv1d(*args, **kwargs)
  nn.init.kaiming_normal_(layer.weight)
  return layer


@torch.jit.script
def silu(x):
  return x * torch.sigmoid(x)


class DiffusionEmbedding(nn.Module):
  def __init__(self, max_steps):
    super().__init__()
    self.register_buffer('embedding', self._build_embedding(max_steps), persistent=False)
    self.projection1 = Linear(128, 512)
    self.projection2 = Linear(512, 512)

  def forward(self, diffusion_step):
    if diffusion_step.dtype in [torch.int32, torch.int64]:
      x = self.embedding[diffusion_step]
    else:
      x = self._lerp_embedding(diffusion_step)
    x = self.projection1(x)
    x = silu(x)
    x = self.projection2(x)
    x = silu(x)
    return x

  def _lerp_embedding(self, t):
    low_idx = torch.floor(t).long()
    high_idx = torch.ceil(t).long()
    low = self.embedding[low_idx]
    high = self.embedding[high_idx]
    return low + (high - low) * (t - low_idx)

  def _build_embedding(self, max_steps):
    steps = torch.arange(max_steps).unsqueeze(1)  # [T,1]
    dims = torch.arange(64).unsqueeze(0)          # [1,64]
    table = steps * 10.0**(dims * 4.0 / 63.0)     # [T,64]
    table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
    return table


class SpectrogramUpsampler(nn.Module):
  def __init__(self, n_mels):
    super().__init__()
    self.conv1 = ConvTranspose2d(1, 1, [3, 32], stride=[1, 16], padding=[1, 8])
    self.conv2 = ConvTranspose2d(1, 1,  [3, 32], stride=[1, 16], padding=[1, 8])

  def forward(self, x):
    x = torch.unsqueeze(x, 1)
    x = self.conv1(x)
    x = F.leaky_relu(x, 0.4)
    x = self.conv2(x)
    x = F.leaky_relu(x, 0.4)
    x = torch.squeeze(x, 1)
    return x


class ResidualBlock(nn.Module):
  def __init__(self, n_mels, residual_channels, dilation, uncond=True):
    '''
    :param n_mels: inplanes of conv1x1 for spectrogram conditional
    :param residual_channels: audio conv
    :param dilation: audio conv dilation
    :param uncond: disable spectrogram conditional
    '''
    super().__init__()
    # self.pos_encoder = PositionalEncoding(2*residual_channels, 24)
    self.dilated_conv = Conv1d(residual_channels, 2 * residual_channels, 3, padding=dilation, dilation=dilation)
    self.diffusion_projection = Linear(512, residual_channels)
    self.con_projection = Conv1d(int(residual_channels), 2*residual_channels,  3, padding=dilation, dilation=dilation)
    if not uncond: # conditional model
      self.conditioner_projection = Conv1d(n_mels, 2 * residual_channels, 1)
    else: # unconditional model
      self.conditioner_projection = None

    self.output_projection = Conv1d(residual_channels, 2 * residual_channels, 1)

  def forward(self, x, diffusion_step,cond_info, conditioner=None):
    # assert (conditioner is None and self.conditioner_projection is None) or \
        #    (conditioner is not None and self.conditioner_projection is not None)

    diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
    y = x + diffusion_step
    # y=self.pos_encoder(y)
    if self.conditioner_projection is None: # using a unconditional model
        y = self.dilated_conv(y)
    else:
    #   conditioner = self.conditioner_projection(conditioner)
    #   y = self.dilated_conv(y) + conditioner
        y = self.dilated_conv(y)

    if cond_info is not None:
      # cond_info=cond_info.permute(0, 2, 1)
      cond_info=self.con_projection(cond_info)
      y=y+cond_info
    else:
      y=y
    gate, filter = torch.chunk(y, 2, dim=1)
    y = torch.sigmoid(gate) * torch.tanh(filter)

    y = self.output_projection(y)
    residual, skip = torch.chunk(y, 2, dim=1)
    return (x + residual) / sqrt(2.0), skip


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model).cuda()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1)]

# Assuming your input tensor is input_tensor of shape [batch_size, seq_len, feature_dim]
# d_model = 6
# max_len = 24
# pos_encoder = PositionalEncoding(d_model, max_len)
# input_tensor = torch.randn(1, 24, 6)  # Example input tensor
# output_tensor = pos_encoder(input_tensor)

class DiffWave(nn.Module):
  def __init__(self,
            residual_channels,
            n_mels=1,
            dcl=10,
            residual_layers=10,
            noise_schedule=np.linspace(1e-4, 0.05, 500).tolist(),
            
            unconditional=False):
    super().__init__()
 
    self.input_projection = Conv1d(residual_channels, 2*residual_channels, 1)
    self.con_projection = Conv1d(residual_channels, 2*residual_channels, 1)
    self.pos_encoder = PositionalEncoding(residual_channels, 24)
    self.diffusion_embedding = DiffusionEmbedding(len(noise_schedule))
    if unconditional: # use unconditional model
      self.spectrogram_upsampler = None
    else:
      self.spectrogram_upsampler = SpectrogramUpsampler(n_mels)

    self.residual_layers = nn.ModuleList([
        # ResidualBlock(n_mels, residual_channels, 2**(i % dcl), uncond=unconditional)
        ResidualBlock(n_mels, 2*residual_channels, 2**(i % 5), uncond=unconditional)
        for i in range(residual_layers)
    ])

    self.reverse_residual_layers = nn.ModuleList([
        # ResidualBlock(n_mels, residual_channels, 2**(i % dcl), uncond=unconditional)
        ResidualBlock(n_mels, 2*residual_channels, 2**(i % 5), uncond=unconditional)
        for i in range(residual_layers)
    ])

    self.skip_projection = Conv1d(2*residual_channels, 2*residual_channels, 1)
    self.output_projection = Conv1d(2*residual_channels, residual_channels, 1)
    nn.init.zeros_(self.output_projection.weight)

  def get_randmask(self, x):
    mask = torch.ones_like(x)
    mask[:, ::2, :] = 0  
    reverse_mask=1-mask
    # masked_input = x * mask
    # reverse_mask = x * reverse_mask
    masked_input = mask
    reverse_mask =reverse_mask
    return masked_input, reverse_mask
  
  def generate_mask_like(self,x):
    mask_ratio=0.5
    mask = torch.rand_like(x) < mask_ratio
    return mask.float()
  
  def forward(self, audio, diffusion_step,train,spectrogram=None):
    # assert (spectrogram is None and self.spectrogram_upsampler is None) or \
    #        (spectrogram is not None and self.spectrogram_upsampler is not None)
    audio=self.pos_encoder(audio)
    x = audio.permute(0, 2, 1)
    
    x = self.input_projection(x)
    x = F.relu(x)# mask,reverse_mask=self.get_randmask(x)
    diffusion_step = self.diffusion_embedding(diffusion_step)
    mask_full = torch.ones_like(x)
    mask_none = torch.zeros_like(x)
    mask = self.generate_mask_like(x)
    mask_input=x
    
    if train:

      skip = None
      for layer in self.residual_layers:
        mask_x, skip_connection = layer(mask_input, diffusion_step,mask=False, spectrogram=None)
        skip = skip_connection if skip is None else skip_connection + skip

      mask_x = skip / sqrt(len(self.residual_layers))
      res=mask_x*mask+x*(1-mask)



      mskip = None
      for layer in self.residual_layers:
        reverse_x, mskip_connection = layer(res, diffusion_step,mask=False, spectrogram=None)
        mskip = skip_connection if mskip is None else mskip_connection + mskip

      reverse_x = mskip / sqrt(len(self.residual_layers))

      x=(x+reverse_x)/sqrt(2.0)
    else:
      # diffusion_step = self.diffusion_embedding(diffusion_step)
      # if self.spectrogram_upsampler: # use conditional model
      #   spectrogram = self.spectrogram_upsampler(spectrogram)
      # x_start=x

      skip = None
      for layer in self.residual_layers:
        x, skip_connection = layer(x, diffusion_step,mask=False, spectrogram=None)
        skip = skip_connection if skip is None else skip_connection + skip

      x = skip / sqrt(len(self.residual_layers))

    # x,_=self.get_randmask(x)
    # _,reverse_x=self.get_randmask(reverse_x)

    # x=(x+reverse_x)
   
    x = self.skip_projection(x)
    x = F.relu(x)
    x = self.output_projection(x).permute(0,2,1)

    return x
