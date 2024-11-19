import itertools
import math
import numbers
import torch
from torch import nn
from torch.autograd import Function
import torch.nn.functional as F

from rankaae.models.transformer import TransformerEnergyPositionPredictor


class Swish(nn.Module):
    def __init__(self, num_parameters, init=0.25, dtype=torch.float32):
        super(Swish, self).__init__()
        init = init * 4.0
        self.beta = nn.Parameter(
            torch.full((num_parameters,), fill_value=(init - 1.0), dtype=dtype), 
            requires_grad=True)
        self.init_value = init
    
    def forward(self, x):
        new_shape = [1, self.beta.size(0)] + [1] * (len(x.size()) - 2)
        ex_beta = 1.0 + self.beta.reshape(new_shape)
        x = x * F.sigmoid(ex_beta * x)
        return x
    
    def extra_repr(self):
        s = f'{self.beta.size(0)}, init={self.init_value}'
        return s
    

def activation_function(name, num_parameters, init=0.25):
    act_cls_dict = {"ReLU": nn.ReLU, "PReLU": nn.PReLU, "Sigmoid": nn.Sigmoid,
                    "Swish": Swish, "Hardswish": nn.Hardswish, "ReLU6": nn.ReLU6,
                    "SiLU": nn.SiLU, "Mish": nn.Mish, "Hardsigmoid": nn.Hardsigmoid,
                    "Softmax": nn.Softmax, "Softplus": nn.Softplus}
    act_cls = act_cls_dict[name]
    params = {}
    if name in ["PReLU", "Swish"]:
        params["num_parameters"] = num_parameters
        params["init"] = init
    if name in ["Softmax"]:
        params["dim"] = 1
    if name == 'Softplus':
        params["beta"] = 2
    act_obj = act_cls(**params)
    return act_obj


class GradientReversalLayer(Function):

    @staticmethod
    def forward(ctx, x, beta):
        ctx.beta = beta
        return x

    @staticmethod
    def backward(ctx, grad_output):
        """If `beta` is none, than this layer does nothing.
        """
        grad_input = grad_output.clone()
        if ctx.beta is not None:
            grad_input = -grad_input * ctx.beta
        return grad_input, None


class GaussianSmoothing(nn.Module):
    def __init__(self, channels, kernel_size, sigma, dim=2, device='cpu'):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ],
            indexing='ij'
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel.to(device))
        self.groups = channels

    def forward(self, x):
        """
        Apply gaussian filter to input.
        Arguments:
            x (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """

        if len(x.size()) - 2 == 1:
            conv = nn.functional.conv1d
        elif len(x.size()) - 2 == 2:
            conv = nn.functional.conv2d
        elif len(x.size()) - 2 == 3:
            conv = nn.functional.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(
                    len(x.size()) - 2)
            )
        return conv(x, weight=self.weight, groups=self.groups)


class FCEncoder(nn.Module):
    
    """
    Fully connected layers for encoder.
    """

    def __init__(
        self, 
        nstyle=5, 
        dim_in=256, 
        n_layers=3,
        hidden_size=64,
        activation='Swish'):
        super(FCEncoder, self).__init__()

        self.dim_in = dim_in
        sequential_layers = [nn.Linear(dim_in, hidden_size, bias=True)] # first layer
        for _ in range(n_layers-1):
            sequential_layers.extend([
                activation_function(activation, num_parameters=hidden_size, init=1.0),
                nn.BatchNorm1d(hidden_size, affine=False),
                nn.Linear(hidden_size, hidden_size, bias=True)])
        sequential_layers.extend([
            activation_function(activation, num_parameters=hidden_size, init=1.0),
            nn.Linear(hidden_size, nstyle, bias=True),
            nn.BatchNorm1d(nstyle, affine=False),])
            # add this batchnorm layer to make sure the output is standardized.
        self.main = nn.Sequential(*sequential_layers)
        self.pre_trained = False

    def forward(self, spec):
        if self.pre_trained:
            self.eval()
        z_gauss = self.main(spec)
        # need to call spec.unsqueeze to accomondate the channel sizes.

        return z_gauss
    
    def get_training_parameters(self):
        return self.parameters()


class FCDecoder(nn.Module):

    def __init__(
        self, 
        nstyle=5, 
        debug=False, 
        dim_out=256, 
        activation='Swish',
        last_layer_activation='ReLu', 
        n_layers=3,
        hidden_size=64
    ):
        super(FCDecoder, self).__init__()

        ll_act = activation_function(last_layer_activation, num_parameters=dim_out)

        sequential_layers = [nn.Linear(nstyle, hidden_size, bias=True)] # the first layer.
        for _ in range(n_layers-2):
            sequential_layers.extend([ # the n layers in the middle
                activation_function(activation, num_parameters=hidden_size, init=1.0),
                nn.BatchNorm1d(hidden_size, affine=False),
                nn.Linear(hidden_size, hidden_size, bias=True)])
        sequential_layers.extend([ # the last layer
            activation_function(activation, num_parameters=hidden_size, init=1.0),
            nn.BatchNorm1d(hidden_size, affine=False),
            nn.Linear(hidden_size, dim_out, bias=True),
            ll_act])  
        self.main = nn.Sequential(*sequential_layers)
        
        self.dim_out = dim_out
        self.nstyle = nstyle
        self.debug = debug
        self.pre_trained = False

    def forward(self, z_gauss):
        if self.pre_trained:
            self.eval()
        spec = self.main(z_gauss)
        return spec
    
    def get_training_parameters(self):
        return self.parameters()


class TwoHotGenerator(nn.Module):
    def __init__(self, gate_window):
        super(TwoHotGenerator, self).__init__()
        self.gate_window = gate_window
    
    def forward(self, spec):
        spec_size = spec.size()
        assert len(spec_size) == 2
        spec = torch.clamp(spec, min=0.0, 
                           max=(self.gate_window - 1.0 - 1.0E-6))
        lower_pos = torch.floor(spec)
        i_lower_pos = lower_pos.to(torch.long)
        
        grid = torch.meshgrid([torch.arange(dim_size, device=spec.device) 
                               for dim_size in spec_size], indexing='ij')
        lower_indices = (grid[0], i_lower_pos, grid[1]) 
        upper_indices = (grid[0], i_lower_pos + 1, grid[1])
        upper_frac = spec - i_lower_pos.to(torch.float32)

        twohot = torch.zeros([spec_size[0], self.gate_window, spec_size[1]], 
                             dtype=torch.float32, requires_grad=False, device=spec.device)
        twohot[lower_indices] = 1.0 - upper_frac
        twohot[upper_indices] = upper_frac
        return twohot


class ExLayers(nn.Module):
    def __init__(self,
                 dim_in: int,
                 dim_out: int,
                 gate_window=13,
                 n_polynomial_order=3,
                 n_polynomial_points=10,
                 padding_mode='stretch',
                 transformer_d_model=2,
                 transformer_hidden_size=64,
                 transformer_nheads=1,
                 transformer_activation='relu',
                 transformer_dropout=0.1,
                 transformer_layers=3):
        super(ExLayers, self).__init__()
        self.compute_padding_params(dim_in, dim_out, gate_window, padding_mode)

        assert transformer_nheads > 0
        self.ene_pos = TransformerEnergyPositionPredictor(
            n_grid=dim_in,
            d_model=transformer_d_model,
            nhead=transformer_nheads,
            dim_feedforward=transformer_hidden_size,
            nlayers=transformer_layers,
            dropout=transformer_dropout,
            batch_first=True,
            activation=transformer_activation)
        self.two_hot_generator = TwoHotGenerator(gate_window)
        self.gate_window = gate_window

        uw = torch.eye(gate_window, dtype=torch.float32, requires_grad=False)[:, None, :]
        self.register_buffer('unfold_weights', uw)

        self.polynomial_weights = nn.Parameter(torch.zeros(
            [n_polynomial_order + 1, 1, n_polynomial_points, 1]), 
            requires_grad=True)
        self.polynomial_interp_size = [dim_out, 1]
        exponents = torch.arange(n_polynomial_order + 1.0, dtype=torch.float)[None, :, None]
        self.register_buffer('exponents', exponents)

    def forward(self, spec):
        ep = self.ene_pos(spec)
        ep = (ep * self.gate_window / 2.0) + (self.gate_window / 2.0)
        ene_sel = self.two_hot_generator(ep)
        spec = self.pad_spectra(spec)
        spec = F.conv1d(spec, self.unfold_weights)
        spec = (spec * ene_sel).sum(dim=1)
        
        pw = F.interpolate(self.polynomial_weights, size=self.polynomial_interp_size, 
                           mode='bicubic', align_corners=True)
        pw = pw.squeeze(dim=-1).squeeze(dim=1)
        d_spec = (torch.pow(spec[:, None, :], self.exponents) * pw[None, :, :]).sum(dim=1)
        spec = spec + d_spec
        return spec

    def compute_padding_params(self, dim_in, dim_out, gate_window, padding_mode):
        if padding_mode == 'stretch':
            pre_dim_out = dim_out + (gate_window - 1)
            self.num_pads = None
        else:
            pre_dim_out = dim_out
            left_padding = gate_window // 2
            self.num_pads = (left_padding, (gate_window - 1) - left_padding)
        self.stretch_scale_factor = pre_dim_out / dim_in
        self.padding_mode = padding_mode

    def pad_spectra(self, spec):
        spec = spec[:, None, :, None]
        if self.stretch_scale_factor != 1.0:
            spec = nn.functional.interpolate(spec, 
                scale_factor=self.stretch_scale_factor, mode='bicubic', align_corners=True)
        spec = spec.squeeze(dim=3)
        if self.padding_mode != 'stretch':
            pm = self.padding_mode.replace('zeros', 'constant')
            spec = F.pad(spec, self.num_pads, mode=pm)
        return spec


class ExEncoder(nn.Module):
    def __init__(self,
                 dim_in: int,
                 enclosing_encoder: FCEncoder,
                 gate_window=13,
                 n_polynomial_order=3,
                 n_polynomial_points=10,
                 padding_mode='stretch',
                 transformer_d_model=2,
                 transformer_hidden_size=64,
                 transformer_nheads=1,
                 transformer_activation='relu',
                 transformer_dropout=0.1,
                 transformer_layers=3):
        super(ExEncoder, self).__init__()
        self.ex_layers = ExLayers(
            dim_in=dim_in, 
            dim_out=enclosing_encoder.dim_in,
            gate_window=gate_window, 
            n_polynomial_order=n_polynomial_order,
            n_polynomial_points=n_polynomial_points,
            padding_mode=padding_mode,
            transformer_d_model=transformer_d_model,
            transformer_hidden_size=transformer_hidden_size,
            transformer_nheads=transformer_nheads,
            transformer_activation=transformer_activation,
            transformer_dropout=transformer_dropout,
            transformer_layers=transformer_layers)
        self.enclosing_encoder = enclosing_encoder

    def forward(self, spec):
        spec = self.ex_layers(spec)
        z_gauss = self.enclosing_encoder(spec)
        return z_gauss
    
    def get_training_parameters(self):
        return self.ex_layers.parameters()


class ExDecoder(nn.Module):
    def __init__(self,
                 dim_out: int,
                 enclosing_decoder: FCDecoder,
                 gate_window=13,
                 n_polynomial_order=3,
                 n_polynomial_points=10,
                 padding_mode='stretch',
                 transformer_d_model=2,
                 transformer_hidden_size=64,
                 transformer_nheads=1,
                 transformer_activation='relu',
                 transformer_dropout=0.1,
                 transformer_layers=3):
        super(ExDecoder, self).__init__()
        self.ex_layers = ExLayers(
            dim_in=enclosing_decoder.dim_out, 
            dim_out=dim_out,
            gate_window=gate_window,
            n_polynomial_order=n_polynomial_order,
            n_polynomial_points=n_polynomial_points,
            padding_mode=padding_mode,
            transformer_d_model=transformer_d_model,
            transformer_hidden_size=transformer_hidden_size,
            transformer_nheads=transformer_nheads,
            transformer_activation=transformer_activation,
            transformer_dropout=transformer_dropout,
            transformer_layers=transformer_layers)
        self.enclosing_decoder = enclosing_decoder
        self.nstyle = enclosing_decoder.nstyle

    def forward(self, z_gauss):
        spec = self.enclosing_decoder(z_gauss)
        spec = self.ex_layers(spec)
        return spec

    def get_training_parameters(self):
        return self.ex_layers.parameters()


class DiscriminatorFC(nn.Module):
    def __init__(self, hiden_size=64, nstyle=5, noise=0.1, layers=3, activation='Swish'):
        super(DiscriminatorFC, self).__init__()
        
        sequential_layers = [nn.Linear(nstyle, hiden_size, bias=True)]
        for _ in range(layers-2):
            sequential_layers.extend([
                activation_function(activation, num_parameters=hiden_size, init=1.0),
                nn.BatchNorm1d(hiden_size, affine=False),
                nn.Linear(hiden_size, hiden_size, bias=True)])
        sequential_layers.extend([
            activation_function(activation, num_parameters=hiden_size, init=1.0),
            nn.BatchNorm1d(hiden_size, affine=False),
            nn.Linear(hiden_size, 1, bias=True)])
        self.main = nn.Sequential(*sequential_layers)
        
        self.nstyle = nstyle
        self.noise = noise
    
    def forward(self, x, beta):
        if self.training:
            x = x + self.noise * torch.randn_like(x, requires_grad=False)
        reverse_feature = GradientReversalLayer.apply(x, beta)
        out = self.main(reverse_feature)
        return out
