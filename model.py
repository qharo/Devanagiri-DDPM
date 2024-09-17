import torch
import torch.nn as nn

def get_time_embedding(time_steps, temb_dim):
    r"""
    Convert time steps tensor into an embedding using the
    sinusoidal time embedding formula
    :param time_steps: 1D tensor of length batch size
    :param temb_dim: Dimension of the embedding
    :return: BxD embedding representation of B time steps
    """
    assert temb_dim % 2 == 0, "time embedding dimension must be divisible by 2"
    
    # factor = 10000^(2i/d_model)
    factor = 10000 ** ((torch.arange(
        start=0, end=temb_dim // 2, dtype=torch.float32, device=time_steps.device) / (temb_dim // 2))
    )
    
    # pos / factor
    # timesteps B -> B, 1 -> B, temb_dim
    t_emb = time_steps[:, None].repeat(1, temb_dim // 2) / factor
    t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)
    return t_emb

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim, 
                 block_type="mid", rescale=False, n_heads=4, n_layers=1):
        super().__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.block_type = block_type

        self._init_resnets(out_channels, in_channels)
        self._init_attention(out_channels, in_channels)

        self.t_emb_layers = nn.ModuleList([
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(t_emb_dim, out_channels)
            )
            for _ in range(self.n_layers+1 if block_type=="mid" else self.n_layers)
        ])

        if self.block_type == "down":
            self.down_sample_conv = nn.Conv2d(out_channels, out_channels,
                                          4, 2, 1) if rescale else nn.Identity()
        if self.block_type == "up":
            self.up_sample_conv = nn.ConvTranspose2d(in_channels // 2, in_channels // 2,
                                                 4, 2, 1) if rescale else nn.Identity()    
    
     # Initialize ResNet Sub-Blocks
    def _init_resnets(self, out_channels, in_channels):
        n_layers = self.n_layers + 1 if self.block_type == "mid" else self.n_layers

        self.resnet_one = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, in_channels if i == 0 else out_channels),
                    nn.SiLU(),
                    nn.Conv2d(in_channels if i == 0 else out_channels, out_channels,
                              kernel_size=3, stride=1, padding=1),
                )
                for i in range(n_layers)
            ]
        )
    
        self.resnet_two = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, out_channels),
                    nn.SiLU(),
                    nn.Conv2d(out_channels, out_channels, 
                        kernel_size=3, stride=1, padding=1),
                )
                for _ in range(n_layers)
            ]
        )

        self.res_input = nn.ModuleList(
            [
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
                for i in range(n_layers)
            ]
        )
    
    # Initialize Attention Sub-Blocks
    def _init_attention(self, out_channels, in_channels):
  
        self.attention_norms = nn.ModuleList(
            [
                nn.GroupNorm(8, out_channels)
                for _ in range(self.n_layers)
            ]
        )
        
        self.attentions = nn.ModuleList(
            [
                nn.MultiheadAttention(out_channels, self.n_heads, batch_first=True)
                for _ in range(self.n_layers)
            ]
        )
    
    def _apply_resnets(self, out, layer):
        resnet_input = out
        out = self.resnet_one[layer](out)
        out = out + self.t_emb_layers[layer](t_emb)[:, :, None, None]
        out = self.resnet_two[layer](out)
        return out + self.residual_input[layer](resnet_input)

    def _apply_attention(self, out, layer):
        batch_size, channels, h, w = out.shape
        in_attn = out.reshape(batch_size, channels, h * w)
        in_attn = self.attention_norms[layer](in_attn)
        in_attn = in_attn.transpose(1, 2)
        out_attn, _ = self.attentions[layer](in_attn, in_attn, in_attn)
        out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
        return out + out_attn

    def forward(self, x, t_emb, out_down=None):
        if self.block_type == "up":
            x = self.up_sample_conv(x)
            x = torch.cat([x, out_down], dim=1)

        out = x

        if block_type == "mid":
            self._apply_resnets(out, 0)
            for i in range(self.num_layers):
                out = self._apply_attention(out, i)
                out = self._apply_resnets(out, i + 1)
        else:
            for i in range(self.num_layers):
                out = self._apply_resnets(out, i)
                out = self._apply_attention(out, i)

        if block_type == "down":
            out = self.down_sample_conv(out)

        return out

class UNet(nn.Module):    
    def __init__(self, model_config):
        super().__init__()
        im_channels = model_config['im_channels']
        self.down_channels = model_config['down_channels']
        self.mid_channels = model_config['mid_channels']
        self.t_emb_dim = model_config['time_emb_dim']
        self.down_sample = model_config['down_sample']
        self.num_down_layers = model_config['num_down_layers']
        self.num_mid_layers = model_config['num_mid_layers']
        self.num_up_layers = model_config['num_up_layers']
        
        assert self.mid_channels[0] == self.down_channels[-1]
        assert self.mid_channels[-1] == self.down_channels[-2]
        assert len(self.down_sample) == len(self.down_channels) - 1
        
        # Initial projection from sinusoidal time embedding
        self.t_proj = nn.Sequential(
            nn.Linear(self.t_emb_dim, self.t_emb_dim),
            nn.SiLU(),
            nn.Linear(self.t_emb_dim, self.t_emb_dim)
        )

        self.up_sample = list(reversed(self.down_sample))
        self.conv_in = nn.Conv2d(im_channels, self.down_channels[0], kernel_size=3, padding=(1, 1))
        
        self.downs = nn.ModuleList([])
        for i in range(len(self.down_channels)-1):
            self.downs.append(Block(self.down_channels[i], self.down_channels[i+1], self.t_emb_dim, block_type="down", rescale=self.down_sample[i], n_layers=self.num_down_layers))


        self.mids = nn.ModuleList([])
        for i in range(len(self.mid_channels)-1):
            self.mids.append(Block(self.mid_channels[i], self.mid_channels[i+1], self.t_emb_dim,block_type="mid", n_layers=self.num_mid_layers))

        self.ups = nn.ModuleList([])
        for i in reversed(range(len(self.down_channels)-1)):
            self.ups.append(Block(self.down_channels[i] * 2, self.down_channels[i-1] if i != 0 else 16, self.t_emb_dim, block_type="up", rescale=self.down_sample[i], n_layers=self.num_up_layers))        

        self.norm_out = nn.GroupNorm(8, 16)
        self.conv_out = nn.Conv2d(16, im_channels, kernel_size=3, padding=1)



    def forward(self, x, t):
        # Shapes assuming downblocks are [C1, C2, C3, C4]
        # Shapes assuming midblocks are [C4, C4, C3]
        # Shapes assuming downsamples are [True, True, False]
        # B x C x H x W
        out = self.conv_in(x)
        # B x C1 x H x W
        
        # t_emb -> B x t_emb_dim
        t_emb = get_time_embedding(torch.as_tensor(t).long(), self.t_emb_dim)
        t_emb = self.t_proj(t_emb)
        
        down_outs = []
        
        for idx, down in enumerate(self.downs):
            down_outs.append(out)
            out = down(out, t_emb)
        # down_outs  [B x C1 x H x W, B x C2 x H/2 x W/2, B x C3 x H/4 x W/4]
        # out B x C4 x H/4 x W/4
            
        for mid in self.mids:
            out = mid(out, t_emb)
        # out B x C3 x H/4 x W/4
        
        for up in self.ups:
            down_out = down_outs.pop()
            out = up(out, down_out, t_emb)
            # out [B x C2 x H/4 x W/4, B x C1 x H/2 x W/2, B x 16 x H x W]
        out = self.norm_out(out)
        out = nn.SiLU()(out)
        out = self.conv_out(out)
        # out B x C x H x W
        return out
