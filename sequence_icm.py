"""Utilities for training."""
# pylint: disable=no-member
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F


class SequenceICM:

    def __init__(self, mechanisms, discriminator_data, discriminator_mechanisms, args):
        super(SequenceICM, self).__init__()
        self.mechs = mechanisms
        self.d_data = discriminator_data
        self.d_mech = discriminator_mechanisms
        self.data_loss_fn = nn.BCEWithLogitsLoss(reduction="mean")
        self.mech_loss_fn = nn.CrossEntropyLoss()
        all_mech_params = []
        for m in self.mechs:
            all_mech_params += list(m.parameters())
        self.mech_opt = torch.optim.Adam(
            all_mech_params + list(self.d_data.parameters())
        )
        self.d_opt = torch.optim.Adam(self.d_mech.parameters())

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
        if verbose:
            print('mech_out_combined', mech_out_combined.shape)
        d_data_scores = self.d_data(mech_out_combined)
        data_target = torch.full((self.bs,), 1.0, device=self.device).unsqueeze(dim=1)
        if verbose:
            print('d_data_scores', d_data_scores.shape)
            print('data_target', data_target.shape)
        data_loss = self.data_loss_fn(d_data_scores, data_target)
        # compute loss for identifiability
        if verbose:
            print('data', data.shape)
            print('data_mech', data_mech.shape)
            print('mech_out_combined', mech_out_combined.shape)
#         data_pairs = torch.cat(
#             (torch.unsqueeze(data_mech, dim=1), torch.unsqueeze(mech_out_combined, dim=1)),
#             dim=1,
#         )
        data_pairs = torch.cat(
            (data_mech, mech_out_combined),
            dim=1,
        )
        d_mech_logits = self.d_mech(data_pairs)
        d_mech_label = torch.arange(0, self.n_mech, device=self.device)
        d_mech_label = d_mech_label.repeat_interleave(self.bs // self.n_mech)
        if verbose:
            print('d_mech_label', d_mech_label.shape)
            print(d_mech_label)
            print('d_mech_logits', d_mech_logits.shape)
        mech_loss = self.mech_loss_fn(d_mech_logits, d_mech_label)
        # backwards for mechanisms
#         total_loss = data_loss + 2.0 * mech_loss
        total_loss = data_loss
        total_loss.backward()
        self.mech_opt.step()

        ##################
        # train data discriminator
        ##################
        data_disc = torch.split(data_disc, self.bs // self.n_mech)
        mech_out = []
        for m, d in zip(self.mechs, data_disc):
            mech_out.append(m(d))
        # compute loss for distribution matching
        mech_out_combined = torch.cat(mech_out, dim=0)
        fake_scores = self.d_data(mech_out_combined)
        real_scores = self.d_data(data_mech)
        real_target = torch.full((self.bs,), 1.0, device=self.device).unsqueeze(dim=1)
        fake_target = torch.full((self.bs,), 0.0, device=self.device).unsqueeze(dim=1)
        real_loss = self.data_loss_fn(real_scores, real_target.detach())
        fake_loss = self.data_loss_fn(fake_scores, fake_target.detach())
        total_loss = real_loss + fake_loss
        total_loss.backward()
        self.d_opt.step()

        return {
            'fake discriminator loss': data_loss.detach().item(),
            'disentangle loss': mech_loss.detach().item(),
            'real discriminator loss': real_loss.detach().item()
        }
