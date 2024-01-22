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
        dropout_rate=0.2, 
        nstyle=5, 
        dim_in=256, 
        n_layers=3,
        hidden_size=64):
        super(FCEncoder, self).__init__()

        self.dim_in = dim_in
        sequential_layers = [ # first layer
            nn.Linear(dim_in, hidden_size),
            Swish(num_parameters=hidden_size, init=1.0),
            nn.BatchNorm1d(hidden_size, affine=False),
            nn.Dropout(p=dropout_rate),
            
        ]

        for _ in range(n_layers-2):
            sequential_layers.extend(
                [   nn.Linear(hidden_size, hidden_size),
                    Swish(num_parameters=hidden_size, init=1.0),
                    nn.BatchNorm1d(hidden_size, affine=False),
                    nn.Dropout(dropout_rate),
                ]
            )

        sequential_layers.extend( # last layer
            [
                nn.Linear(hidden_size, nstyle),
                nn.BatchNorm1d(nstyle, affine=False)
                # add this batchnorm layer to make sure the output is standardized.
            ]
        )

        self.main = nn.Sequential(*sequential_layers)

    def forward(self, spec):
        
        z_gauss = self.main(spec)
        # need to call spec.unsqueeze to accomondate the channel sizes.

        return z_gauss
    
    def get_training_parameters(self):
        return self.parameters()

class FCDecoder(nn.Module):

    def __init__(
        self, 
        dropout_rate=0.2, 
        nstyle=5, 
        debug=False, 
        dim_out=256, 
        last_layer_activation='ReLu', 
        n_layers=3,
        hidden_size=64
    ):
        super(FCDecoder, self).__init__()

        if last_layer_activation == 'ReLu':
            ll_act = nn.ReLU()
        elif last_layer_activation == 'Swish':
            ll_act = Swish(num_parameters=dim_out)
        elif last_layer_activation == 'Softplus':
            ll_act = nn.Softplus(beta=2)
        else:
            raise ValueError(
                f"Unknow activation function \"{last_layer_activation}\", please use one available in Pytorch")

        sequential_layers = [ # the first layer.
                nn.Linear(nstyle, hidden_size),
                Swish(num_parameters=hidden_size, init=1.0),
                nn.BatchNorm1d(hidden_size, affine=False),
                nn.Dropout(p=dropout_rate),
        ]

        for _ in range(n_layers-2):
            sequential_layers.extend( # the n layers in the middle
                [
                    nn.Linear(hidden_size, hidden_size),
                    Swish(num_parameters=hidden_size, init=1.0),
                    nn.BatchNorm1d(hidden_size, affine=False),
                    nn.Dropout(p=dropout_rate),
                ]
            )
        sequential_layers.extend( # the last layer
            [
                nn.Linear(hidden_size, dim_out),
                ll_act,
            ]
        )  

        self.main = nn.Sequential(*sequential_layers)
        
        self.dim_out = dim_out
        self.nstyle = nstyle
        self.debug = debug

    def forward(self, z_gauss):
        spec = self.main(z_gauss)
        return spec
    
    def get_training_parameters(self):
        return self.parameters()


class ExEncoder(nn.Module):
    def __init__(self,
                 dim_in: int,
                 dropout_rate: float,
                 enclosing_encoder: FCEncoder):
        super(ExEncoder, self).__init__()
        self.ex_layers = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(dim_in, enclosing_encoder.dim_in))
        self.enclosing_encoder = enclosing_encoder

    def forward(self, spec):
        x = self.ex_layers(spec)
        z_gauss = self.enclosing_encoder(x)
        return z_gauss
    
    def get_training_parameters(self):
        return self.ex_layers.parameters()


class ExDecoder(nn.Module):
    def __init__(self,
                 dim_out: int,
                 dropout_rate: float,
                 enclosing_decoder: FCDecoder):
        super(ExDecoder, self).__init__()
        self.ex_layers = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(enclosing_decoder.dim_out, dim_out))   
        self.enclosing_decoder = enclosing_decoder
        self.nstyle = enclosing_decoder.nstyle

    def forward(self, z_gauss):
        x = self.enclosing_decoder(z_gauss)
        spec = self.ex_layers(x)
        return spec
    
    def get_training_parameters(self):
        return self.ex_layers.parameters()

class DiscriminatorCNN(nn.Module):
    def __init__(self, hiden_size=64, channels=2, kernel_size=5, dropout_rate=0.2, nstyle=5, noise=0.1):
        super(DiscriminatorCNN, self).__init__()

        self.pre = nn.Sequential(
            nn.Linear(nstyle, hiden_size),
            Swish(num_parameters=hiden_size, init=1.0)
        )

        self.main = nn.Sequential(
            nn.BatchNorm1d(1, affine=False),
            nn.Conv1d(1, channels, kernel_size=kernel_size, padding=(
                kernel_size-1)//2, padding_mode='replicate'),
            Swish(num_parameters=channels, init=1.0),

            nn.BatchNorm1d(channels, affine=False),
            nn.Conv1d(channels, channels, kernel_size=kernel_size,
                      padding=(kernel_size-1)//2, padding_mode='replicate'),
            Swish(num_parameters=channels, init=1.0),

            nn.BatchNorm1d(channels, affine=False),
            nn.Conv1d(channels, channels, kernel_size=kernel_size,
                      padding=(kernel_size-1)//2, padding_mode='replicate'),
            Swish(num_parameters=channels, init=1.0),

            nn.BatchNorm1d(channels, affine=False),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=(kernel_size-1)//2,
                      padding_mode='replicate'),
            Swish(num_parameters=channels, init=1.0),

            nn.BatchNorm1d(channels, affine=False),
            nn.Conv1d(channels, 1, kernel_size=kernel_size, padding=(
                kernel_size-1)//2, padding_mode='replicate'),
            Swish(num_parameters=1, init=1.0)
        )

        self.post = nn.Sequential(
            nn.BatchNorm1d(hiden_size, affine=False),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hiden_size, 1),
        )

        self.nstyle = nstyle
        self.noise = noise

    def forward(self, x, beta):
        if self.training:
            x = x + self.noise * torch.randn_like(x, requires_grad=False)
        x = GradientReversalLayer.apply(x, beta)
        x = self.pre(x)
        x = x.unsqueeze(dim=1)
        x = self.main(x)
        x = x.squeeze(dim=1)
        out = self.post(x)
        return out


class DiscriminatorFC(nn.Module):
    def __init__(self, hiden_size=64, dropout_rate=0.2, nstyle=5, noise=0.1, layers=3):
        super(DiscriminatorFC, self).__init__()
        
        sequential_layers = [
            nn.Linear(nstyle, hiden_size),
            Swish(num_parameters=hiden_size, init=1.0),
            nn.Dropout(p=dropout_rate),
        ]
        for _ in range(layers-2):
            sequential_layers.extend(
                [
                    nn.Linear(hiden_size, hiden_size),
                    Swish(num_parameters=hiden_size, init=1.0),
                    nn.Dropout(p=dropout_rate),
                ]
            )
        sequential_layers.extend(
            [
                nn.Linear(hiden_size, 1),
            ]
        )
        self.main = nn.Sequential(*sequential_layers)
        
        self.nstyle = nstyle
        self.noise = noise
    
    def forward(self, x, beta):
        if self.training:
            x = x + self.noise * torch.randn_like(x, requires_grad=False)
        reverse_feature = GradientReversalLayer.apply(x, beta)
        out = self.main(reverse_feature)
        return out

class DummyDualAAE(nn.Module):
    def __init__(self, use_cnn_dis, cls_encoder, cls_decoder):
        super(DummyDualAAE, self).__init__()
        self.encoder = cls_encoder()
        self.decoder = cls_decoder()
        self.discriminator = DiscriminatorCNN() if use_cnn_dis else DiscriminatorFC()

    def forward(self, x):
        z = self.encoder(x)
        x2 = self.decoder(z)
        is_gau = self.discriminator(z, 0.3)
        return x2, is_gau
        
    def forward(self, specs_in):
        seg_selector = self.warp_indexer(self.grid_indices)
        specs_warped = (specs_in[:, self.seg_source_indices] * seg_selector[None, ...]).sum(dim=-1)
        specs_ws = specs_warped * self.k + self.b
        return specs_ws

