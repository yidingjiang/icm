"""Utilities for training."""
# pylint: disable=no-member
import torch
import numpy as np
import torch.nn.functional as F


def initialize_experts(experts, data, args):
    """Function for initialize a list of experts.

    Args:
        experts (list): a list of all expert
        optimizers (list): a list of optimizer for each expert
        data (DataLoader): data loader for all the training data
        args : argparse object
    """
    print("======Initializing experts to identity on target data===\n")
    loss = torch.nn.MSELoss(reduction="mean")
    optimizers = []
    for e in experts:
        optimizers.append(torch.optim.Adam(e.parameters()))
    for i, e, o in zip(np.arange(len(experts)), experts, optimizers):
        initialize_single_expert(e, o, data, loss, i, args)
    print("\n======Finished initializing experts===\n")


def initialize_single_expert(expert, optimizer, data, loss, index, args):
    """Initialize a single object.

    Args:
        expert (Expert): a single expert
        optimizer : Pytorch optimizer for the expert
        data (DataLoader): data loader for the training data
        loss : type of loss function used for training the initialization
        index (int): unique index for the current expert
        args : argparse object
    """
    expert.train()
    for epoch in range(args.num_initialize_epoch):
        total_loss = 0
        n_batch = 0
        for batch in data:
            n_batch += 1
            _, x = batch
            x = x.to(args.device)
            x_hat = expert(x)
            l2_diff = loss(x_hat, x)
            total_loss += l2_diff.item()
            optimizer.zero_grad()
            l2_diff.backward()
            optimizer.step()
            # Loss
            mean_loss = total_loss / n_batch
            if mean_loss < args.min_initialization_loss:
                print(
                    "initialization: expert {} epoch {} loss {:.4f}".format(
                        index, epoch + 1, mean_loss
                    )
                )
                print("------------below threshold----------")
                return


def train_icm(experts, expert_opt, discriminator, discriminator_opt, data, args):
    """Execute the ICM training.

    Args:
        experts (Expert): a list of all experts
        expert_opt (list): a list of all optmizers for each expert
        discriminator (Discriminator): the discriminator
        discriminator_opt : the optimizer for training the discriminator
        data (DataLoader): pytorch dataloader for the training data
        args : argparse object
    """
    loss = torch.nn.BCEWithLogitsLoss(reduction="mean")
    discriminator.train()
    _ = [e.train() for e in experts]
    torch.autograd.set_detect_anomaly(True)

    for idx, batch in enumerate(data):
        x_src, x_tgt = batch
        x_src, x_tgt = x_src.to(args.device), x_tgt.to(args.device)

        if args.no_source_target:
            x_src = x_tgt

        # D expert pass
        discriminator_opt.zero_grad()

        exp_out, exp_score = [], []
        loss_exp_d = 0
        labels = torch.full((args.batch_size,), 0.0, device=args.device).unsqueeze(dim=1)
        for e in experts:
            out = e(x_tgt)
            # print(out.size())
            score = discriminator(out.detach())
            exp_out.append(out)
            exp_score.append(score)
            loss_exp_d += loss(score, labels)
        loss_exp_d /= len(experts)
        # loss_exp_d.backward()

        # D discriminator pass
        score = discriminator(x_src)
        # labels.fill_(1)
        labels = torch.full((args.batch_size,), 1.0, device=args.device).unsqueeze(dim=1)
        total_loss = loss(score, labels.detach()) + loss_exp_d

        # combined and back pass for discriminator
        total_loss.backward()
        discriminator_opt.step()

        # train experts
        exp_out = [torch.unsqueeze(out, 1) for out in exp_out]
        exp_out = torch.cat(exp_out, dim=1)
        exp_score = torch.cat(exp_score, dim=1)
        winning_idx = exp_score.argmax(dim=1)

        per_expert_winning_num = []

        for i, e in enumerate(experts):
            selected_idx = winning_idx.eq(i).nonzero().squeeze(dim=-1)
            n_samples = selected_idx.size(0)
            per_expert_winning_num.append(n_samples)
            if n_samples > 0:
                expert_opt[i].zero_grad()
                # samples = exp_out[selected_idx, i]
                samples = e(x_tgt[selected_idx])
                score = discriminator(samples)
                labels = torch.full((n_samples,), 1.0, device=args.device)
                labels = labels.unsqueeze(dim=1)
                loss_exp = loss(score, labels)
                loss_exp.backward(retain_graph=True)
                expert_opt[i].step()

        if idx % 100 == 0:
            print("Discriminator expert Loss: {:.4f}".format(loss_exp_d))
            print(
                "Discriminator discriminator Loss: {:.4f}".format(
                    total_loss - loss_exp_d
                )
            )
            print("Per expert winning num: {}\n".format(per_expert_winning_num))


def loss_hinge_dis(dis_fake, dis_real):
    loss_real = torch.mean(F.relu(1. - dis_real))
    loss_fake = torch.mean(F.relu(1. + dis_fake))
    return loss_real, loss_fake


def loss_hinge_gen(dis_fake):
    loss = -torch.mean(dis_fake)
    return loss


def train_gan(experts, expert_opt, discriminator, discriminator_opt, data, args):
    loss = torch.nn.BCELoss(reduction="mean")
    expert, expert_opt = experts[0], expert_opt[0]
    discriminator.train()
    expert.train()
    for idx, batch in enumerate(data):
        _, x_tgt = batch

        # D
        noise = torch.Tensor(np.random.normal(size=[args.batch_size, 2]) * args.noise_scale)
        out = expert(noise)
        gen_score = discriminator(out)
        real_score = discriminator(x_tgt)
        gen_label = torch.full((args.batch_size,), 0.0, device=args.device).unsqueeze(dim=1)
        real_label = torch.full((args.batch_size,), 1.0, device=args.device).unsqueeze(dim=1)
        # d_loss_fake = loss(gen_score, gen_label.detach())
        # d_loss_real = loss(real_score, real_label.detach())
        d_loss_real, d_loss_fake = loss_hinge_dis(gen_score, real_score)
        total_loss = d_loss_fake + d_loss_real
        discriminator_opt.zero_grad()
        total_loss.backward(retain_graph=True)
        discriminator_opt.step()

        # G
        noise = torch.Tensor(np.random.normal(size=[args.batch_size, 2]) * args.noise_scale)
        out = expert(noise)
        gen_score = discriminator(out)
        gen_label = torch.full((args.batch_size,), 1.0, device=args.device).unsqueeze(dim=1)
        # gen_loss = loss(gen_score, gen_label.detach())
        gen_loss = loss_hinge_gen(gen_score)
        expert_opt.zero_grad()
        gen_loss.backward(retain_graph=True)
        expert_opt.step()

        if idx % args.print_iterval == 0:
            print('Iteration {}'.format(idx))
            print("Discriminator fake Loss: {:.4f}".format(d_loss_fake))
            print("Discriminator real Loss: {:.4f}".format(d_loss_real))
            print("Generator Loss: {:.4f} \n".format(gen_loss))
