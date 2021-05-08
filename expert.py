"""Definition of models."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
from layers import SNLinear
from layers import SNConv2d


class Expert(nn.Module):
    """Single expert.

    Args:
        args: argparse object
    """

    def __init__(self, args):
        super(Expert, self).__init__()
        self.config = args
        self.model = self.build()

    def build(self):
        layers = [
            SNLinear(self.config.input_shape, 64),
            nn.ReLU(),
            SNLinear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.config.input_shape),
        ]
        return nn.Sequential(*layers)

    def forward(self, input):
        out = self.model(input)
        return out


class AffineExpert(Expert):
    """Expert that uses linear functions."""

    def build(self):
        layers = [
            nn.Linear(self.config.input_shape, self.config.input_shape),
        ]
        return nn.Sequential(*layers)


class TranslationExpert(AffineExpert):
    def build(self):
        layers = [
            nn.Linear(self.config.input_shape, self.config.input_shape),
        ]
        identity = np.float32([[1.0, 0.0], [0.0, 1.0]])
        layers[0].weight = nn.parameter.Parameter(
            torch.tensor(identity), requires_grad=False
        )
        return nn.Sequential(*layers)


class ConvolutionExpert(Expert):
    """Expert for images."""

    def build(self):
        conv_layer = SNConv2d if self.config.use_sn else nn.Conv2d
        layers = [
            conv_layer(self.config.input_shape[-1], 16, 3, 1, padding=1),
            nn.ReLU(),
            conv_layer(16, 32, 3, 1, padding=1),
            nn.ReLU(),
            conv_layer(32, 16, 3, 1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 1, 1, padding=0),
            nn.Sigmoid(),
        ]
        return nn.Sequential(*layers)


class STNExpert(nn.Module):

    def __init__(self, args):
        super(STNExpert, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.sigmoid = nn.Sigmoid()

        self.conv1 = nn.Conv2d(args.input_shape[-1], 32, 3, 1, padding=1)
        self.lrelu1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(32, args.input_shape[-1], kernel_size=3, padding=1)
    
        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        # theta = torch.clamp(theta, -1, 1)
        grid = F.affine_grid(theta, x.size(), align_corners=True)
        x = F.grid_sample(x, grid, align_corners=True)

        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)
        return x

        # # Perform the usual forward pass
        # x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # x = x.view(-1, 320)
        # x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        # x = self.fc2(x)
        # return F.log_softmax(x, dim=1)


class AffineExpert(nn.Module):
    """ """
    def __init__(self, args):
        super(AffineExpert, self).__init__()
        # self.Sigma = nn.Parameter(torch.eye(3), requires_grad=True)
        # self.Mu = nn.Parameter(torch.zeros(3), requires_grad=True)
        self.Mu = nn.Parameter(torch.zeros(200), requires_grad=True)
        # self.Mu2 = nn.Parameter(torch.zeros(3), requires_grad=True)
        # self.Mu2 = nn.Parameter(torch.zeros(3), requires_grad=True)
        self.projection = nn.Sequential(
            nn.Linear(200, 200),
            nn.ReLU(True),
            nn.Linear(200, 200),
            nn.ReLU(True),
            nn.Linear(200, 3),
        )
        # self.conv1 = nn.Conv2d(args.input_shape[-1], 32, 3, 1, padding=1)
        # self.lrelu1 = nn.LeakyReLU()
        # self.conv2 = nn.Conv2d(32, args.input_shape[-1], kernel_size=3, padding=1)
        # self.Mu = nn.Parameter(torch.zeros(3) + torch.randn(3))
        self.sigmoid = nn.Sigmoid()

    def translateRotate(self, x, mu):
        bs, _, w, h = x.size()
        # z = torch.randn(bs,3,device=x.device,dtype=x.dtype)@self.Sigma + self.Mu
        z = torch.zeros((bs,3),device=x.device,dtype=x.dtype) + mu
        # z = torch.tanh(0.1*z)
        # Build affine matrices for random translation of each image
        affineMatrices = torch.zeros(bs,2,3,device=x.device,dtype=x.dtype)
        affineMatrices[:,0,0] = z[:,2].cos().detach()
        affineMatrices[:,0,1] = -z[:,2].sin().detach()
        affineMatrices[:,1,0] = z[:,2].sin().detach()
        affineMatrices[:,1,1] = z[:,2].cos().detach()
        affineMatrices[:,:2,2] = z[:,:2]/(.5*w+.5*h)

        # affineMatrices[:,:2,2] = z[:,:2]
        # affineMatrices[:,:2,2] = (z[:,:2]+0.5*w)/(.5*w+.5*h)
        # affineMatrices[:,:2,2] = (z[:,:2]+10.)/(.5*w+.5*h)
        affineMatrices = affineMatrices

        flowgrid = F.affine_grid(affineMatrices, size = x.size(),align_corners=False)
        x_out = F.grid_sample(x, flowgrid, align_corners=False)
        return x_out

    def forward(self, x):
        # x_max = torch.amax(x, dim=(1, 2, 3), keepdim=True)
        # print(self.Mu)
        # mu = torch.clamp(self.Mu, -0.01, 0.01)
        mu = self.projection(self.Mu)
        out = self.translateRotate(x, mu)
        # for _ in range(10):
        #     out = self.translateRotate(out, self.Mu)
        return out
        # out_max = torch.amax(x, dim=(1, 2, 3), keepdim=True).detach()
        # print(self.Mu)
        # return out / out_max * x_max
        # return self.sigmoid(self.conv2(self.conv1(out)))
        # out = self.lrelu1(self.conv1(out))
        # return self.sigmoid(x + self.conv2(out))


class ExpertFilter(nn.Module):
    def __init__(self, args):
        super(ExpertFilter, self).__init__()
        conv_layer = SNConv2d if args.use_sn else nn.Conv2d
        layers = [
            conv_layer(args.input_shape[-1], 64, 3, 1, padding=1),
            nn.LeakyReLU(),
            conv_layer(64, 64, 3, 1, padding=1),
            nn.LeakyReLU(),
            conv_layer(64, 32, 3, 1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, args.input_shape[-1], 1, 1, padding=0),
            nn.Sigmoid(),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


#####################################################

def expm(A,rtol=1e-4):
    """ assumes A has shape (bs,d,d)
        returns exp(A) with shape (bs,d,d) """
    I = torch.eye(A.shape[-1],device=A.device,dtype=A.dtype)[None].repeat(A.shape[0],1,1)
    return odeint(lambda t, x: A@x,I,torch.tensor([0.,1.]).to(A.device,A.dtype), rtol=rtol)[-1]


class AffineExpertV2(nn.Module):
    """docstring for MLPAug"""
    def __init__(self, args, gen_scale=10., trans_scale=0.1,
                 epsilon=1e-3, random=False):
        super(AffineExpertV2, self).__init__()
        self.random = random
        self.trans_scale = trans_scale

        self.width = nn.Parameter(torch.zeros(6))
        self.softplus = torch.nn.Softplus()
        self.g0 = None
        self.std_batch_size = None

    def set_width(self, vals):
        self.width.data = vals
        
    def transform(self, x):
        bs, _, w, h = x.size()
        if self.random:
            weights = torch.rand(bs, 6)
        else:
            weights = torch.zeros((bs, 6))
        weights = weights.to(x.device, x.dtype)
        width = self.softplus(self.width)
        weights = weights * width - width.div(2.)

        generators = self.generate(weights)

        ## exponential map

        affine_matrices = expm(generators.cpu()).to(weights.device)

        flowgrid = F.affine_grid(affine_matrices[:, :2, :], size = x.size(),
                                 align_corners=True)
        x_out = F.grid_sample(x, flowgrid, align_corners=True)
        return x_out

    def generate(self, weights):
        """
        return the sum of the scaled generator matrices
        """
        bs = weights.shape[0]

        if self.g0 is None or self.std_batch_size != bs:
            self.std_batch_size = bs

            ## tx
            self.g0 = torch.zeros(3, 3, device=weights.device)
            self.g0[0, 2] = 1. * self.trans_scale
            self.g0 = self.g0.unsqueeze(-1).expand(3,3, bs)

            ## ty
            self.g1 = torch.zeros(3, 3, device=weights.device)
            self.g1[1, 2] = 1. * self.trans_scale
            self.g1 = self.g1.unsqueeze(-1).expand(3,3, bs)

            self.g2 = torch.zeros(3, 3, device=weights.device)
            self.g2[0, 1] = -1.
            self.g2[1, 0] = 1.
            self.g2 = self.g2.unsqueeze(-1).expand(3,3, bs)

            self.g3 = torch.zeros(3, 3, device=weights.device)
            self.g3[0, 0] = 1.
            self.g3[1, 1] = 1.
            self.g3 = self.g3.unsqueeze(-1).expand(3,3, bs)

            self.g4 = torch.zeros(3, 3, device=weights.device)
            self.g4[0, 0] = 1.
            self.g4[1, 1] = -1.
            self.g4 = self.g4.unsqueeze(-1).expand(3,3, bs)

            self.g5 = torch.zeros(3, 3, device=weights.device)
            self.g5[0, 1] = 1.
            self.g5[1, 0] = 1.
            self.g5 = self.g5.unsqueeze(-1).expand(3,3, bs)

        out_mat = weights[:, 0] * self.g0
        out_mat += weights[:, 1] * self.g1
        out_mat += weights[:, 2] * self.g2
        out_mat += weights[:, 3] * self.g3
        out_mat += weights[:, 4] * self.g4
        out_mat += weights[:, 5] * self.g5

        # transposes just to get everything right
        return out_mat.transpose(0, 2).transpose(2, 1)

    def forward(self, x):
        return self.transform(x)
