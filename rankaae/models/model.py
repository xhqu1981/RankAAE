import math
import numbers
import torch
from torch import nn
from torch.autograd import Function
import torch.nn.functional as F


class Swish(nn.Module):
    def __init__(self, num_parameters, init=1.0):
        super(Swish, self).__init__()
        self.beta = nn.Parameter(
            torch.full((num_parameters,), fill_value=init, dtype=torch.float32), 
            requires_grad=True)
    
    def forward(self, x):
        new_shape = [1, self.beta.size(0)] + [1] * (len(x.size()) - 2)
        ex_beta = self.beta.reshape(new_shape)
        x = x * F.sigmoid(ex_beta * x)
        return x


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
        hidden_size=64):
        super(FCEncoder, self).__init__()

        self.dim_in = dim_in
        sequential_layers = [nn.Linear(dim_in, hidden_size, False)] # first layer
        for _ in range(n_layers-2):
            sequential_layers.extend([
                nn.BatchNorm1d(hidden_size, affine=True),
                Swish(num_parameters=hidden_size, init=1.0),
                nn.Linear(hidden_size, hidden_size, bias=False)])
        sequential_layers.extend([ # last layer
            nn.BatchNorm1d(hidden_size, affine=True),
            Swish(num_parameters=hidden_size, init=1.0),
            nn.Linear(hidden_size, nstyle, bias=False),
            nn.BatchNorm1d(nstyle, affine=False)])
            # add this batchnorm layer to make sure the output is standardized.
        self.main = nn.Sequential(*sequential_layers)

    def forward(self, spec):
        
        z_gauss = self.main(spec)
        # need to call spec.unsqueeze to accomondate the channel sizes.

        return z_gauss
    
    def get_training_parameters(self):
        return self.parameters()


def build_activation_function(num_parameters, activation_name):
        if activation_name == 'ReLU':
            ll_act = nn.ReLU()
        elif activation_name == 'PReLU':
            ll_act = nn.PReLU(num_parameters=num_parameters)
        elif activation_name == 'Swish':
            ll_act = Swish(num_parameters=num_parameters)
        elif activation_name == 'Softplus':
            ll_act = nn.Softplus(beta=2)
        else:
            raise ValueError(
                f"Unknow activation function \"{activation_name}\", please use one available in Pytorch")
                
        return ll_act


class FCDecoder(nn.Module):

    def __init__(
        self, 
        nstyle=5, 
        debug=False, 
        dim_out=256, 
        last_layer_activation='ReLu', 
        n_layers=3,
        hidden_size=64
    ):
        super(FCDecoder, self).__init__()

        ll_act = build_activation_function(dim_out, last_layer_activation)

        sequential_layers = [nn.Linear(nstyle, hidden_size, bias=False)] # the first layer.
        for _ in range(n_layers-2):
            sequential_layers.extend([ # the n layers in the middle
                nn.BatchNorm1d(hidden_size, affine=True),
                Swish(num_parameters=hidden_size, init=1.0),
                nn.Linear(hidden_size, hidden_size, bias=False)])
        sequential_layers.extend([ # the last layer
            nn.BatchNorm1d(hidden_size, affine=True),
            Swish(num_parameters=hidden_size, init=1.0),
            nn.Linear(hidden_size, dim_out),
            ll_act])  
        self.main = nn.Sequential(*sequential_layers)
        
        self.dim_out = dim_out
        self.nstyle = nstyle
        self.debug = debug

    def forward(self, z_gauss):
        spec = self.main(z_gauss)
        return spec
    
    def get_training_parameters(self):
        return self.parameters()
    

class ExLayers(nn.Module):
    def __init__(self,
                 dim_in: int,
                 dim_out: int,
                 gate_window=13,
                 n_exlayers=1,
                 n_gate_layers=5,
                 n_channels=13,
                 last_layer_activation='Softplus',
                 padding_mode='stretch',
                 energy_noise=0.1):
        super(ExLayers, self).__init__()

        if padding_mode == 'stretch':
            pre_dim_out = dim_out + (gate_window - 1)
            self.padding_mode = 'no'
        elif padding_mode == 'zeros':
            pre_dim_out = dim_out
            self.padding_mode = 'constant'
        else:
            pre_dim_out = dim_out
            self.padding_mode = padding_mode
        left_padding = gate_window // 2
        self.num_pads = (left_padding, (gate_window - 1) - left_padding) 
        self.scale_factor = pre_dim_out / dim_in
        self.energy_noise = energy_noise

        pe = torch.arange(dim_out, dtype=torch.float32, requires_grad=False) + 1
        self.register_buffer("position_embedding", pe[:, None])
        gate_layers = [nn.Linear(1, gate_window, bias=False)]
        assert n_gate_layers >= 2
        for _ in range(n_gate_layers-2):
            gate_layers.extend([
                nn.BatchNorm1d(gate_window, affine=True),
                Swish(num_parameters=gate_window, init=1.0),
                nn.Linear(gate_window, gate_window, bias=False)])
        gate_layers.extend([
            nn.BatchNorm1d(gate_window, affine=True),
            Swish(num_parameters=gate_window, init=1.0),
            nn.Linear(gate_window, gate_window, bias=True),
            nn.Softmax(dim=1)])  
        self.gate_weights = nn.Sequential(**gate_layers)
        uw = torch.eye(gate_window, dtype=torch.float32, requires_grad=False)[:, None, :]
        self.register_buffer('upend_weights', uw)

        assert n_exlayers > 0
        if n_exlayers == 1:
            intensity_layers = [
                nn.Conv1d(1, 1, kernel_size=1, bias=True)]
        else:
            intensity_layers = [
                nn.Conv1d(1, n_channels, kernel_size=1, bias=False)]
        for _ in range(n_exlayers - 2):
            intensity_layers.extend([
                nn.BatchNorm1d(n_channels, affine=True),
                Swish(num_parameters=n_channels, init=1.0),
                nn.Conv1d(n_channels, n_channels, kernel_size=1, bias=False)])
        if n_exlayers >= 2:
            intensity_layers.extend([
                nn.BatchNorm1d(n_channels, affine=True),
                Swish(num_parameters=n_channels, init=1.0),
                nn.Conv1d(n_channels, 1, kernel_size=1, bias=True)])
        if last_layer_activation:
            ll_act = build_activation_function(1, last_layer_activation)
            intensity_layers.append(ll_act)
        self.intensity_adjuster = nn.Sequential(*intensity_layers) 

    def forward(self, spec):
        spec = nn.functional.interpolate(spec[:, None, :], 
            scale_factor=self.scale_factor, mode='linear', align_corners=True)
        if self.padding_mode != 'no':
            spec = F.pad(spec, self.num_pads, mode=self.padding_mode)
        spec = self.intensity_adjuster(spec)
        spec = F.conv1d(spec, self.upend_weights)
        if self.training:
            pe = self.position_embedding + torch.randn_like(self.position_embedding) * self.energy_noise
        else:
            pe = self.position_embedding
        sel_weights = self.gate_weights(pe).T[None, ...]
        spec = (spec * sel_weights).sum(dim=1)
        spec = spec.squeeze(dim=1)
        return spec


class ExEncoder(nn.Module):
    def __init__(self,
                 dim_in: int,
                 enclosing_encoder: FCEncoder,
                 kernel_size=13,
                 hidden_kernel_size=1,
                 n_exlayers=1,
                 n_channels=13,
                 last_layer_activation='Softplus',
                 padding_mode='stretch'):
        super(ExEncoder, self).__init__()
        self.ex_layers = ExLayers(dim_in=dim_in, dim_out=enclosing_encoder.dim_in,
            gate_window=kernel_size, hidden_kernel_size=hidden_kernel_size,
            n_exlayers=n_exlayers, n_channels=n_channels, 
            last_layer_activation=last_layer_activation, 
            padding_mode=padding_mode)
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
                 kernel_size=13,
                 hidden_kernel_size=1,
                 n_exlayers=1,
                 n_channels=13,
                 last_layer_activation='Softplus',
                 padding_mode='stretch'):
        super(ExDecoder, self).__init__()
        self.ex_layers = ExLayers(dim_in=enclosing_decoder.dim_out, dim_out=dim_out,
            gate_window=kernel_size, hidden_kernel_size=hidden_kernel_size, 
            n_exlayers=n_exlayers, n_channels=n_channels, 
            last_layer_activation=last_layer_activation, 
            padding_mode=padding_mode)
        self.enclosing_decoder = enclosing_decoder
        self.nstyle = enclosing_decoder.nstyle

    def forward(self, z_gauss):
        spec = self.enclosing_decoder(z_gauss)
        spec = self.ex_layers(spec)
        return spec

    def get_training_parameters(self):
        return self.ex_layers.parameters()


class DiscriminatorFC(nn.Module):
    def __init__(self, hiden_size=64, nstyle=5, noise=0.1, layers=3):
        super(DiscriminatorFC, self).__init__()
        
        sequential_layers = [nn.Linear(nstyle, hiden_size, bias=False)]
        for _ in range(layers-2):
            sequential_layers.extend([
                nn.BatchNorm1d(hiden_size, affine=True),
                Swish(num_parameters=hiden_size, init=1.0),
                nn.Linear(hiden_size, hiden_size, bias=False)])
        sequential_layers.extend([
            nn.BatchNorm1d(hiden_size, affine=True),
            Swish(num_parameters=hiden_size, init=1.0),
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
