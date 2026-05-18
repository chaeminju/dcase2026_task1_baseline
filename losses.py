import torch
import torch.nn as nn
import numpy as np
from einops import rearrange

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

        self.cross_entropy = nn.CrossEntropyLoss(label_smoothing=0.01)

    def forward(self, logits, labels):
        return self.cross_entropy(logits, labels)

class UnagiContrastiveLoss(nn.Module):
    def __init__(self, views):
        super().__init__()
        self.views = views

    def combine_views(self, *views):
        return torch.stack(list(views), dim=1)

    def expand_target(self, Y):
        return rearrange(Y, "(b v) -> b v", v=self.views)
    
def weighted_logsumexp(mat, axis, weights):
    _max, _ = torch.max(mat, dim=axis, keepdim=True)

    lse = (
        (torch.exp(mat - _max) * weights)
        .sum(dim=axis, keepdim=True)
        .log()
        + _max
    )

    return lse.squeeze(axis)

class ContrastiveLoss(UnagiContrastiveLoss):
    def __init__(
        self,
        views,
        type="l_spread",  # sup_con, sim_clr, l_attract, l_spread
        temp=0.5,
        pos_in_denom=False,  # as per dan, false by default
        log_first=True,  # TODO (ASN): should this be true (false originally)
        a_lc=1.0,
        a_spread=1.0,
        lc_norm=False,
        use_labels=True,
        clip_pos=1.0,
        pos_in_denom_weight=1.0,
    ):
        super().__init__(views)
        self.temp = temp
        self.log_first = log_first
        self.a_lc = a_lc
        self.a_spread = a_spread
        self.pos_in_denom = pos_in_denom
        self.lc_norm = lc_norm
        self.use_labels = use_labels
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.clip_pos = clip_pos
        self.pos_in_denom_weight = pos_in_denom_weight

        if type == "sup_con":
            print(f"Using {type} contrastive loss function")
            self.a_spread = 0
            # self.pos_in_denom = False  # this isn't doing anything
        # elif type == "l_attract":
        #     print(f"Using {type} contrastive loss function")
        #     self.a_spread = 0
        #     self.pos_in_denom = False  # working
        elif type == "l_repel":
            print(f"Using {type} contrastive loss function")
            self.a_spread = 1
            self.a_lc = 0
        elif type == "sim_clr":
            print(f"Using {type} contrastive loss function")
            self.a_spread = 0
            self.a_lc = 1
            self.use_labels = False

    def forward(self, *args):
        inputs = args[:-1]
        Y = args[-1]
        x, labels = self.combine_views(*inputs), self.expand_target(Y)

        # x has shape batch * num views * dimension
        # labels has shape batch * num views
        b, nViews, d = x.size()
        vs = torch.split(x, 1, dim=1)  # images indexed by view
        if not self.use_labels:
            labels = torch.full(labels.shape, -1)
        ts = torch.split(labels, 1, dim=1)  # labels indexed by view
        l = 0.0
        pairs = nViews * (nViews - 1) // 2

        for ii in range(nViews):
            vi = vs[ii].squeeze()
            ti = ts[ii].squeeze()

            ti_np = np.array([int(label) for label in ti])
            for jj in range(ii):
                vj = vs[jj].squeeze()

                # num[i,j] is f(xi) * f(xj) / tau, for i,j
                if self.lc_norm:
                    num = (
                        torch.einsum("b d, c d -> b c", vi, vj)
                        .div(self.temp)
                        .div(torch.norm(vi, dim=1) * torch.norm(vj, dim=1))
                    )
                else:
                    num = torch.einsum("b d, c d -> b c", vi, vj).div(self.temp)

                # store the first positive (augmentation of the same view)
                pos_ones = []
                neg_ones = []  # store the first negative
                M_indices = []
                div_factor = []

                for i, cls in enumerate(ti_np):
                    # fall back to SimCLR
                    pos_indices = torch.tensor([i]).to(ti.device)
                    if cls != -1:
                        pos_indices = torch.where(ti == cls)[0]

                    # fall back to SimCLR
                    neg_indices = torch.tensor(
                        [idx for idx in range(ti.shape[0]) if idx != i]
                    ).to(ti.device)

                    if cls != -1:
                        neg_indices = torch.where(ti != cls)[0]

                    all_indices = torch.stack(
                        [
                            torch.cat(
                                (
                                    pos_indices
                                    if self.pos_in_denom
                                    else pos_indices[j : j + 1],
                                    neg_indices,
                                )
                            )
                            for j in range(len(pos_indices))
                        ]
                    )

                    # store all the positive indices
                    pos_ones.append(pos_indices)

                    # store all the negative indices that go up to m
                    neg_ones.append(neg_indices)
                    M_indices.append(all_indices)
                    div_factor.append(len(pos_indices))

                if self.pos_in_denom_weight == 1.0:
                    # denominator for each point in the batch
                    denominator = torch.stack(
                        [
                            # reshape num with an extra dimension, then take the
                            # sum over everything
                            torch.logsumexp(num[i][M_indices[i]], 1).sum()
                            for i in range(len(ti))
                        ]
                    )
                else:
                    # denominator for each Mpoint in the batch
                    denominator = torch.stack(
                        [
                            # reshape num with an extra dimension, then take the
                            # sum over everything
                            weighted_logsumexp(
                                num[i][M_indices[i]],
                                1,
                                torch.tensor(
                                    np.concatenate(
                                        [
                                            np.full(
                                                len(pos_ones[i]),
                                                self.pos_in_denom_weight,
                                            ),
                                            np.ones(len(neg_ones[i])),
                                        ]
                                    )
                                ).to(ti.device),
                            ).sum()
                            for i in range(len(ti))
                        ]
                    )

                if self.clip_pos != 1.0:
                    # numerator
                    numerator = torch.stack(
                        [
                            # sum over all the positives
                            torch.sum(-1 * num[i][pos_ones[i]])
                            #                     -1 * num[i][pos_ones[i]]
                            for i in range(len(ti))
                        ]
                    )
                else:
                    # numerator
                    numerator = torch.stack(
                        [
                            # sum over all the positives
                            torch.sum(
                                -1 * torch.clamp(num[i][pos_ones[i]], max=self.clip_pos)
                            )
                            #                     -1 * num[i][pos_ones[i]]
                            for i in range(len(ti))
                        ]
                    )

                log_prob = numerator + denominator

                if self.a_spread > 0.0:
                    assert self.a_lc + self.a_spread != 0

                    numerator_spread = torch.stack(
                        [
                            -1 * num[i][i]
                            for i in range(len(ti))
                        ]
                    )

                    denominator_spread = torch.stack(
                        [
                            torch.logsumexp(num[i][pos_ones[i]], 0)
                            for i in range(len(ti))
                        ]
                    )

                    log_prob_spread = numerator_spread + denominator_spread

                    a = (
                        self.a_lc * log_prob.div(torch.tensor(div_factor).to(ti.device))
                        + self.a_spread * log_prob_spread
                    ) / (self.a_lc + self.a_spread)
                else:
                    log_prob = log_prob.to(ti.device)
                    a = torch.tensor(self.a_lc).to(ti.device) * log_prob.to(
                        ti.device
                    ).div(torch.tensor(div_factor).to(ti.device))

                l += a.mean()

        out = l / pairs
        return out
    
class HierarchicalContrastiveLoss(nn.Module):
    def __init__(
        self,
        lambda_top=0.03,
        lambda_sub=0.1,
    ):
        super().__init__()

        self.lambda_top = lambda_top
        self.lambda_sub = lambda_sub

        self.top_loss = ContrastiveLoss(
            views=2,
            type="sup_con",
            temp=0.07,
            lc_norm=True
        )

        self.sub_loss = ContrastiveLoss(
            views=2,
            type="l_spread",
            temp=0.07,
            lc_norm=True
        )

    def forward(
        self,
        z1,
        z2,
        top_labels,
        sub_labels
    ):
        top_con = self.top_loss(
            z1,
            z2,
            top_labels
        )

        sub_con = self.sub_loss(
            z1,
            z2,
            sub_labels
        )

        return (
            self.lambda_top * top_con
            + self.lambda_sub * sub_con
        )