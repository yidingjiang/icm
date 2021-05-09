"""Utilities for training."""
# pylint: disable=no-member
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

from discriminator import GaussianSmoothing
from expert import ExpertFilter


def loss_hinge_dis(dis_fake, dis_real):
    loss_real = torch.mean(F.relu(1. - dis_real))
    loss_fake = torch.mean(F.relu(1. + dis_fake))
    return loss_real, loss_fake


def loss_hinge_gen(dis_fake):
    loss = -torch.mean(dis_fake)
    return loss


def difference_loss(input, output, max_diff=50):
    diff = torch.abs(input - output)
    return torch.mean(F.relu(max_diff - torch.sum(diff, axis=[1,2,3])))


def mass_preserving_loss(input, output):
    b, c, w, h = input.shape
    input_mass = torch.sum(input.reshape((b, -1)), dim=-1)
    output_mass = torch.sum(output.reshape((b, -1)), dim=-1)
    diff = input_mass - output_mass
    return torch.mean(diff**2)


class SequenceICM:

    def __init__(self, mechanisms, discriminator_data, discriminator_mechanisms, args, smoothing_raidus=0.5, target_diff=50):
        super(SequenceICM, self).__init__()
        self.mechs = mechanisms
        # self.mech_filter = ExpertFilter(args).to(args.device)
        self.mech_filter = torch.nn.Identity()
        self.dummy_filter = nn.Linear(1, 1)

        self.d_data = discriminator_data
        self.d_mech = discriminator_mechanisms
        self.smoother = GaussianSmoothing(2, 5, smoothing_raidus)

        self.data_loss_fn = nn.BCEWithLogitsLoss(reduction="mean")
        self.mech_loss_fn = nn.CrossEntropyLoss()
        all_mech_params = []
        for m in self.mechs:
            all_mech_params += list(m.parameters())
        # all_mech_params += list(self.mech_filter.parameters())
        # print(all_mech_params)

        # Optimizer
        # self.mech_opt = torch.optim.Adam(
        #     all_mech_params + list(self.d_mech.parameters())
        # )
        self.mech_opt = torch.optim.RMSprop(
            all_mech_params + list(self.d_mech.parameters())
        )
        self.d_opt = torch.optim.Adam(self.d_data.parameters())
        self.mm_opt = torch.optim.Adam(
            list(self.dummy_filter.parameters()) + list(self.mech_filter.parameters()),
            lr=1e-4
        )

        self.target_diff = target_diff
        self.bs = args.batch_size
        self.n_mech = len(self.mechs)
        if self.bs % self.n_mech != 0:
            raise ValueError(
                "Batchsize {} is not divisible by {}".format(self.bs, self.n_mech)
            )
        self.device = args.device

    def train(self, data, verbose=False):
        # set up training mode
        self.d_data.train()
        self.d_mech.train()
        for m in self.mechs:
            m.train()
        # spliting training batch into half
        data = data.to(self.device)
        data_mech, data_disc = data[: self.bs], data[self.bs:]
        # print(data_mech.shape, data_disc.shape)
        ##################
        # train mechanisms and mechanism discriminators
        ##################
        if verbose:
            print('data_mech', data_mech.shape)
        data_mech_split = torch.split(data_mech, self.bs //self.n_mech)
        mech_out = []
        for m, d in zip(self.mechs, data_mech_split):
            if verbose:
                print(d.shape)
            mech_out.append(m(d))
        # compute loss for distribution matching
        mech_out_combined = torch.cat(mech_out, dim=0)
        mech_out_combined_f = self.mech_filter(mech_out_combined)
        # mech_out_combined_f = mech_out_combined
        if verbose:
            print('mech_out_combined', mech_out_combined.shape)
        d_data_scores = self.d_data(mech_out_combined_f)
        data_target = torch.full((self.bs,), 1.0, device=self.device).unsqueeze(dim=1)
        if verbose:
            print('d_data_scores', d_data_scores.shape)
            print('data_target', data_target.shape)
        # data_loss = self.data_loss_fn(d_data_scores, data_target)
        data_loss = loss_hinge_gen(d_data_scores)
        # compute loss for identifiability
        if verbose:
            print('data', data.shape)
            print('data_mech', data_mech.shape)
            print('mech_out_combined', mech_out_combined.shape)

        data_pairs = torch.cat(
            (self.mech_filter(data_mech), mech_out_combined_f),
            dim=1,
        )
        data_pairs = self.smoother(data_pairs)
        d_mech_logits = self.d_mech(data_pairs)
        d_mech_label = torch.arange(0, self.n_mech, device=self.device)
        d_mech_label = d_mech_label.repeat_interleave(self.bs // self.n_mech)
        if verbose:
            print('d_mech_label', d_mech_label.shape)
            print(d_mech_label)
            print('d_mech_logits', d_mech_logits.shape)
        mech_loss = self.mech_loss_fn(d_mech_logits, d_mech_label)
        # backwards for mechanisms
        diff_loss = difference_loss(data_mech, mech_out_combined, self.target_diff)
        mass_loss = mass_preserving_loss(data_mech, mech_out_combined)
        # total_loss = data_loss + mech_loss + diff_loss
        total_loss = data_loss + mech_loss + mass_loss
        # total_loss = data_loss
        # total_loss = mech_loss
        self.mech_opt.zero_grad()
        self.mm_opt.zero_grad()
        total_loss.backward()
        self.mech_opt.step()
        self.mm_opt.step()

        ##################
        # train data discriminator
        ##################
        data_disc_split = torch.split(data_mech, self.bs // self.n_mech)
        mech_out = []
        for m, d in zip(self.mechs, data_disc_split):
            mech_out.append(m(d))
        # compute loss for distribution matching
        mech_out_combined = torch.cat(mech_out, dim=0).detach()
        mech_out_combined = self.mech_filter(mech_out_combined)
        fake_scores = self.d_data(mech_out_combined)
        real_scores = self.d_data(self.mech_filter(data_disc))
        # real_target = torch.full((self.bs,), 1.0, device=self.device).unsqueeze(dim=1)
        # fake_target = torch.full((self.bs,), 0.0, device=self.device).unsqueeze(dim=1)
        # real_loss = self.data_loss_fn(real_scores, real_target.detach())
        # fake_loss = self.data_loss_fn(fake_scores, fake_target.detach())
        # total_loss = real_loss + fake_loss
        real_loss, fake_loss = loss_hinge_dis(fake_scores, real_scores)
        total_loss = fake_loss + real_loss
        self.d_opt.zero_grad()
        self.mm_opt.zero_grad()
        total_loss.backward()
        self.d_opt.step()
        self.mm_opt.step()

        return {
            'fake discriminator loss': fake_loss.detach().item(),
            'disentangle loss': mech_loss.detach().item(),
            'real discriminator loss': real_loss.detach().item(),
            'fake score': torch.mean(fake_scores).item(),
            'real score': torch.mean(real_scores).item(),
            'diff_loss': diff_loss.item(),
        }


class GAN:

    def __init__(self, generator, discriminator, args):
        super(SequenceICM, self).__init__()
        self.g = generator
        self.d = discriminator
        self.loss_fn = torch.nn.BCELoss(reduction="mean")
        self.g_opt = torch.optim.Adam(self.g.parameters())
        self.d_opt = torch.optim.Adam(self.d.parameters())

        self.bs = args.batch_size
        self.device = args.device

    def train(self, data):
        # D
        noise = torch.Tensor(np.random.normal(size=[args.bs, 100]))
        fake = self.g(noise)
        fake_score = self.d(fake)
        real_score = self.d(data)
        d_loss_real, d_loss_fake = loss_hinge_dis(gen_score, real_score)
        total_loss = d_loss_fake + d_loss_real
        self.d_opt.zero_grad()
        total_loss.backward(retain_graph=True)
        self.d_opt.step()

        # G
        noise = torch.Tensor(np.random.normal(size=[args.bs, 100]))
        out = self.g(noise)
        gen_score = self.d(out)
        gen_label = torch.full((args.batch_size,), 1.0, device=args.device).unsqueeze(dim=1)
        gen_loss = loss_hinge_gen(gen_score)
        self.g_opt.zero_grad()
        gen_loss.backward(retain_graph=True)
        self.g_opt.step()

        return {
            'g loss': gen_loss,
            'd loss': total_loss,
            'd_loss_real': d_loss_real,
            'd_loss_fake': d_loss_fake,
        }
