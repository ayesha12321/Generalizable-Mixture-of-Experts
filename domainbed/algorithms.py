# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import sys
from itertools import chain

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import torch.autograd as autograd
from domainbed.lib.misc import (
    random_pairs_of_minibatches, ParamDict, MovingAverage, l2_between_dicts
)
from copy import deepcopy
import copy

sys.path.append('/mnt/lustre/bli/projects/EIL/domainbed')
from . import vision_transformer
# from . import vision_transformer_hybrid
from collections import defaultdict, OrderedDict

try:
    from backpack import backpack, extend
    from backpack.extensions import BatchGrad
except:
    backpack = None

from domainbed import networks
# from domainbed import resnet_variants
import torchvision.models as models

ALGORITHMS = [
    'ERM',
    'Fish',
    'IRM',
    'GroupDRO',
    'Mixup',
    'MLDG',
    'CORAL',
    'MMD',
    'DANN',
    'CDANN',
    'MTL',
    'SagNet',
    'ARM',
    'VREx',
    'RSC',
    'SD',
    'ANDMask',
    'SANDMask',
    'IGA',
    'SelfReg',
    "Fishr",
    'TRM',
    'IB_ERM',
    'IB_IRM',
    'CAD',
    'CondCAD',
    'GMOE',
    'NullExpertMoE'
]


def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]


class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """
    transforms = {}

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Algorithm, self).__init__()
        self.hparams = hparams

    def update(self, minibatches, unlabeled=None):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.

        Admits an optional list of unlabeled minibatches from the test domains,
        when task is domain_adaptation.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError


class MovingAvg:
    def __init__(self, network):
        self.network = network
        self.network_sma = copy.deepcopy(network)
        self.network_sma.eval()
        self.sma_start_iter = 100
        self.global_iter = 0
        self.sma_count = 0

    def update_sma(self):
        self.global_iter += 1
        if self.global_iter >= self.sma_start_iter:
            self.sma_count += 1
            for param_q, param_k in zip(self.network.parameters(), self.network_sma.parameters()):
                param_k.data = (param_k.data * self.sma_count + param_q.data) / (1. + self.sma_count)
        else:
            for param_q, param_k in zip(self.network.parameters(), self.network_sma.parameters()):
                param_k.data = param_q.data


class ERM_SMA(Algorithm, MovingAvg):
    """
    Empirical Risk Minimization (ERM) with Simple Moving Average (SMA) prediction model
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        Algorithm.__init__(self, input_shape, num_classes, num_domains, hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        MovingAvg.__init__(self, self.network)

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        loss = F.cross_entropy(self.network(all_x), all_y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.update_sma()
        return {'loss': loss.item()}

    def predict(self, x):
        self.network_sma.eval()
        return self.network_sma(x)


class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])

        self.network = nn.Sequential(self.featurizer, self.classifier).cuda()
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def predict(self, x):
        return self.network(x)


class GMOE(Algorithm):
    """
    SFMOE
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(GMOE, self).__init__(input_shape, num_classes, num_domains, hparams)
        num_experts = int(self.hparams.get('gmoe_num_experts', 6))
        top_k = int(self.hparams.get('gmoe_top_k', 1))
        self.model = vision_transformer.deit_small_patch16_224(
            pretrained=True,
            num_classes=num_classes,
            moe_layers=['F'] * 8 + ['S', 'F'] * 2,
            mlp_ratio=4.,
            num_experts=num_experts,
            moe_top_k=top_k,
            drop_path_rate=0.1,
            router='cosine_top'
        ).cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams["lr"], weight_decay=self.hparams['weight_decay'])

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        loss = F.cross_entropy(self.predict(all_x), all_y)
        loss_aux_list = []
        for block in self.model.blocks:
            if getattr(block, 'aux_loss') is not None:
                loss_aux_list.append(block.aux_loss)

        loss_aux = torch.tensor(0.0, device=all_x.device)
        if len(loss_aux_list):
            loss_aux = torch.stack(loss_aux_list).sum()

        loss += loss_aux
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item(), 'loss_aux': loss_aux.item()}

    def predict(self, x, forward_feature=False):
        if forward_feature:
            return self.model.forward_features(x)
        else:
            prediction = self.model(x)
            if type(prediction) is tuple:
                return (prediction[0] + prediction[1]) / 2
            else:
                return prediction


class NullExpertMoE(Algorithm):
    """ViT GMoE with one additional null expert and anti-collapse controls."""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(NullExpertMoE, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.num_domains = num_domains
        self.num_real_experts = int(self.hparams.get('gmoe_num_experts', 6))
        self.num_total_experts = self.num_real_experts + 1

        self.model = vision_transformer.deit_small_patch16_224(
            pretrained=True,
            num_classes=num_classes,
            moe_layers=['F'] * 8 + ['S', 'F'] * 2,
            mlp_ratio=4.,
            num_experts=self.num_real_experts,
            moe_top_k=int(self.hparams.get('gmoe_top_k', 2)),
            use_null_moe=True,
            num_domains=num_domains,
            moe_temperature=float(self.hparams.get('null_moe_temperature', 0.1)),
            drop_path_rate=0.1,
            router='cosine_top'
        ).cuda()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.register_buffer('update_count', torch.tensor([0]))

    @staticmethod
    def _stack_mean_or_zero(losses, device):
        if len(losses) == 0:
            return torch.tensor(0.0, device=device)
        return torch.stack(losses).mean()

    @staticmethod
    def _stack_sum_or_zero(losses, device):
        if len(losses) == 0:
            return torch.tensor(0.0, device=device)
        return torch.stack(losses).sum()

    def _collect_moe_outputs(self):
        patch_start = getattr(self.model, 'num_tokens', 1)
        aux_losses = []
        null_probs = []
        top1_routing = []
        domain_logits = []
        token_features = []

        for block in self.model.blocks:
            block_aux = getattr(block, 'aux_loss', None)
            if block_aux is not None:
                aux_losses.append(block_aux)

            block_null = getattr(block, 'null_prob', None)
            if block_null is not None and block_null.shape[1] > patch_start:
                null_probs.append(block_null[:, patch_start:])

            block_top1 = getattr(block, 'top1_routing', None)
            if block_top1 is not None and block_top1.shape[1] > patch_start:
                top1_routing.append(block_top1[:, patch_start:])

            block_domain = getattr(block, 'domain_logits', None)
            if block_domain is not None and block_domain.shape[1] > patch_start:
                domain_logits.append(block_domain[:, patch_start:, :])

            block_features = getattr(block, 'token_features', None)
            if block_features is not None and block_features.shape[1] > patch_start:
                token_features.append(block_features[:, patch_start:, :])

        return aux_losses, null_probs, top1_routing, domain_logits, token_features

    def _compute_domain_losses(self, domain_logits, null_probs, token_features, domain_targets):
        device = domain_targets.device
        domain_cls_losses = []
        spurious_losses = []
        contrastive_losses = []
        adaptive_cap_losses = []
        centered_cap_losses = []
        domain_conf_means = []
        eps = 1e-6
        contrastive_temp = max(float(self.hparams.get('null_moe_contrastive_temp', 0.1)), 1e-6)

        for idx, logits in enumerate(domain_logits):
            batch_size, num_tokens, num_domains = logits.shape
            repeated_targets = domain_targets.view(batch_size, 1).expand(-1, num_tokens).reshape(-1)
            domain_cls_losses.append(F.cross_entropy(logits.reshape(-1, num_domains), repeated_targets))

            domain_conf = F.softmax(logits.detach(), dim=-1).max(dim=-1).values
            domain_conf_means.append(domain_conf.mean())
            if idx < len(null_probs):
                spurious_losses.append((domain_conf * (1.0 - null_probs[idx])).mean())
                adaptive_cap_losses.append((null_probs[idx] * (1.0 - domain_conf)).mean())
                # Standardized spuriousness: zero-mean so the net push on average
                # null usage is 0 (cannot run away), unit-scale so the force does
                # not decay as the domain classifier saturates. Minimizing this
                # raises the correlation between spuriousness and null routing:
                # tokens above average get pushed toward null, those below away.
                centered = (domain_conf - domain_conf.mean()) / (domain_conf.std() + eps)
                centered_cap_losses.append(-(centered * null_probs[idx]).mean())

            if idx < len(token_features):
                features = token_features[idx]
                spurious_weights = domain_conf
                invariant_weights = 1.0 - domain_conf

                spurious_proto = (features * spurious_weights.unsqueeze(-1)).sum(dim=1) / (
                    spurious_weights.sum(dim=1, keepdim=True) + eps
                )
                invariant_proto = (features * invariant_weights.unsqueeze(-1)).sum(dim=1) / (
                    invariant_weights.sum(dim=1, keepdim=True) + eps
                )

                spurious_proto = F.normalize(spurious_proto, p=2, dim=-1)
                invariant_proto = F.normalize(invariant_proto, p=2, dim=-1)
                cosine_sim = (spurious_proto * invariant_proto).sum(dim=-1)
                contrastive_losses.append(((cosine_sim + 1.0) * 0.5 / contrastive_temp).mean())

        domain_cls_loss = self._stack_mean_or_zero(domain_cls_losses, device)
        spurious_loss = self._stack_mean_or_zero(spurious_losses, device)
        contrastive_loss = self._stack_mean_or_zero(contrastive_losses, device)
        adaptive_cap_loss = self._stack_mean_or_zero(adaptive_cap_losses, device)
        centered_cap_loss = self._stack_mean_or_zero(centered_cap_losses, device)
        mean_domain_conf = self._stack_mean_or_zero(domain_conf_means, device)
        return (domain_cls_loss, spurious_loss, contrastive_loss,
                adaptive_cap_loss, centered_cap_loss, mean_domain_conf)

    def _compute_null_usage(self, null_probs, top1_routing, device):
        zero = torch.tensor(0.0, device=device)
        if len(null_probs) == 0:
            return zero, zero, zero, zero, zero, zero, None

        flat_null = torch.cat([null_tensor.reshape(-1) for null_tensor in null_probs], dim=0)
        avg_null = flat_null.mean()
        max_null = flat_null.max()
        pct_over_50 = (flat_null > 0.5).float().mean()
        pct_over_70 = (flat_null > 0.7).float().mean()
        pct_over_90 = (flat_null > 0.9).float().mean()

        null_top1_share = zero
        top1_dist = None
        if len(top1_routing):
            flat_top1 = torch.cat([routing.reshape(-1).long() for routing in top1_routing], dim=0)
            counts = torch.bincount(flat_top1, minlength=self.num_total_experts).float()
            top1_dist = counts / counts.sum().clamp_min(1.0)
            null_top1_share = top1_dist[-1]

        return avg_null, max_null, pct_over_50, pct_over_70, pct_over_90, null_top1_share, top1_dist

    @torch.no_grad()
    def _compute_signal_diagnostics(self, domain_logits, null_probs):
        """Is domain_conf a usable *per-token* spuriousness signal, or just a
        per-image style property? Decomposes its variance into within-image
        (tokens of one image differ) vs between-image (images differ), and
        checks whether the router actually correlates null routing with it.
        Diagnostic only -- no gradients, called on diag steps.
        """
        if len(domain_logits) == 0 or len(null_probs) == 0:
            return None

        conf = torch.cat([
            F.softmax(logits, dim=-1).max(dim=-1).values
            for logits in domain_logits
        ], dim=0)                                    # [B*blocks, N]
        null = torch.cat([n for n in null_probs], dim=0)
        if conf.shape != null.shape:
            return None

        flat_conf, flat_null = conf.reshape(-1), null.reshape(-1)
        # Variance decomposition: does spuriousness vary *inside* an image?
        within_img = conf.std(dim=1).mean()           # spread across tokens of one image
        between_img = conf.mean(dim=1).std()          # spread across images

        cf = flat_conf - flat_conf.mean()
        nl = flat_null - flat_null.mean()
        denom = (cf.norm() * nl.norm()).clamp_min(1e-8)
        corr = (cf * nl).sum() / denom                # does router follow the signal?

        q = torch.tensor([0.10, 0.50, 0.90], device=flat_conf.device)
        c10, c50, c90 = torch.quantile(flat_conf, q).tolist()
        n10, n50, n90 = torch.quantile(flat_null, q).tolist()

        return {
            'conf_mean': flat_conf.mean().item(), 'conf_std': flat_conf.std().item(),
            'conf_p10': c10, 'conf_p50': c50, 'conf_p90': c90,
            'within_img_std': within_img.item(), 'between_img_std': between_img.item(),
            'null_p10': n10, 'null_p50': n50, 'null_p90': n90,
            'corr_conf_null': corr.item(),
        }

    def _maybe_print_null_diagnostics(
            self,
            avg_null,
            max_null,
            pct_over_50,
            pct_over_70,
            pct_over_90,
            null_top1_share,
            top1_dist,
            null_cap_penalty,
            adaptive_cap_loss,
            mean_domain_conf,
            centered_cap_loss=None,
            guard_penalty=None,
            signal_diag=None):
        interval = int(self.hparams.get('null_moe_diag_interval', 500))
        if interval <= 0 or (self.update_count.item() % interval != 0):
            return

        dist_string = 'n/a'
        if top1_dist is not None:
            parts = []
            for expert_idx in range(self.num_real_experts):
                parts.append(f'e{expert_idx}:{top1_dist[expert_idx].item():.3f}')
            parts.append(f'null:{top1_dist[-1].item():.3f}')
            dist_string = ', '.join(parts)

        print(
            '[NullExpertMoE] step={} cap_mode={} avg_null={:.4f} max_null={:.4f} '
            'pct>0.5={:.3f} pct>0.7={:.3f} pct>0.9={:.3f} '
            'null_top1={:.3f} null_cap_penalty={:.6f} '
            'adaptive_cap={:.6f} centered_cap={:.6f} guard={:.6f} '
            'mean_domain_conf={:.4f} top1_dist=[{}]'.format(
                self.update_count.item(),
                str(self.hparams.get('null_moe_cap_mode', 'adaptive')),
                avg_null.item(),
                max_null.item(),
                pct_over_50.item(),
                pct_over_70.item(),
                pct_over_90.item(),
                null_top1_share.item(),
                null_cap_penalty.item(),
                adaptive_cap_loss.item(),
                centered_cap_loss.item() if centered_cap_loss is not None else float('nan'),
                guard_penalty.item() if guard_penalty is not None else float('nan'),
                mean_domain_conf.item(),
                dist_string
            )
        )

        if signal_diag is not None:
            d = signal_diag
            # within_img_std is the decisive number: if it is ~0, domain_conf is a
            # per-image style property and cannot discriminate tokens within an image.
            print(
                '[NullExpertMoE/signal] step={} conf_mean={:.4f} conf_std={:.4f} '
                'conf_p10/50/90={:.3f}/{:.3f}/{:.3f} '
                'within_img_std={:.4f} between_img_std={:.4f} within/between={:.2f} '
                'null_p10/50/90={:.3f}/{:.3f}/{:.3f} corr(conf,null)={:+.4f}'.format(
                    self.update_count.item(),
                    d['conf_mean'], d['conf_std'],
                    d['conf_p10'], d['conf_p50'], d['conf_p90'],
                    d['within_img_std'], d['between_img_std'],
                    d['within_img_std'] / max(d['between_img_std'], 1e-8),
                    d['null_p10'], d['null_p50'], d['null_p90'],
                    d['corr_conf_null'],
                )
            )

    def update(self, minibatches, unlabeled=None):
        del unlabeled

        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        domain_targets = torch.cat([
            torch.full((x.shape[0],), env_idx, dtype=torch.long, device=x.device)
            for env_idx, (x, _) in enumerate(minibatches)
        ])

        logits = self.predict(all_x)
        ce_loss = F.cross_entropy(logits, all_y)

        aux_losses, null_probs, top1_routing, domain_logits, token_features = self._collect_moe_outputs()
        loss_aux = self._stack_sum_or_zero(aux_losses, all_x.device)
        (domain_cls_loss, spurious_loss, contrastive_loss,
         adaptive_cap_loss, centered_cap_loss, mean_domain_conf) = self._compute_domain_losses(
            domain_logits,
            null_probs,
            token_features,
            domain_targets
        )

        avg_null, max_null, pct_over_50, pct_over_70, pct_over_90, null_top1_share, top1_dist = self._compute_null_usage(
            null_probs,
            top1_routing,
            all_x.device
        )

        cap_mode = str(self.hparams.get('null_moe_cap_mode', 'centered'))
        spurious_weight = float(self.hparams.get('null_moe_spurious_weight', 0.1))
        guard_penalty = torch.tensor(0.0, device=all_x.device)
        guard_weight = float(self.hparams.get('null_moe_cap_weight', 1.0))

        if cap_mode == 'fixed':
            null_cap = float(self.hparams.get('null_moe_max_null_ratio', 0.2))
            null_cap_penalty = F.relu(avg_null - null_cap) ** 2
            cap_weight = guard_weight
        elif cap_mode == 'adaptive':
            # Content-aware cap: only penalize null routing on tokens the domain
            # classifier does not consider spurious, so genuinely spurious tokens
            # can reach the null expert for free.
            null_cap_penalty = adaptive_cap_loss
            cap_weight = float(self.hparams.get('null_moe_adaptive_cap_weight', 0.1))
        else:
            # Standardized signal decides which tokens go null. It subsumes
            # spurious_loss -- that term is a one-way push toward null with no
            # counterforce, so keeping both would reintroduce the runaway it
            # caused in 'adaptive' mode. Dropped here deliberately.
            null_cap_penalty = centered_cap_loss
            cap_weight = float(self.hparams.get('null_moe_centered_weight', 0.1))
            spurious_weight = 0.0
            # Dormant guardrail: sits far above normal usage so it never shapes
            # allocation, but prevents a pathological collapse to ~all-null.
            guard_ratio = float(self.hparams.get('null_moe_guard_ratio', 0.35))
            guard_penalty = F.relu(avg_null - guard_ratio) ** 2

        total_loss = (
            ce_loss
            + float(self.hparams.get('null_moe_balance_weight', 0.01)) * loss_aux
            + spurious_weight * spurious_loss
            + float(self.hparams.get('null_moe_contrastive_weight', 0.0)) * contrastive_loss
            + cap_weight * null_cap_penalty
            + guard_weight * guard_penalty
            + float(self.hparams.get('null_moe_domain_weight', 1.0)) * domain_cls_loss
        )

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        self.update_count += 1
        diag_interval = int(self.hparams.get('null_moe_diag_interval', 500))
        signal_diag = None
        if diag_interval > 0 and self.update_count.item() % diag_interval == 0:
            signal_diag = self._compute_signal_diagnostics(domain_logits, null_probs)
        self._maybe_print_null_diagnostics(
            avg_null,
            max_null,
            pct_over_50,
            pct_over_70,
            pct_over_90,
            null_top1_share,
            top1_dist,
            null_cap_penalty,
            adaptive_cap_loss,
            mean_domain_conf,
            centered_cap_loss,
            guard_penalty,
            signal_diag
        )

        return {
            'loss': total_loss.item(),
            'ce_loss': ce_loss.item(),
            'loss_aux': loss_aux.item(),
            'domain_cls_loss': domain_cls_loss.item(),
            'spurious_loss': spurious_loss.item(),
            'contrastive_loss': contrastive_loss.item(),
            'null_cap_penalty': null_cap_penalty.item(),
            'adaptive_cap_loss': adaptive_cap_loss.item(),
            'centered_cap_loss': centered_cap_loss.item(),
            'guard_penalty': guard_penalty.item(),
            'mean_domain_conf': mean_domain_conf.item(),
            'null_avg_prob': avg_null.item(),
            'null_max_prob': max_null.item(),
            'null_top1_share': null_top1_share.item(),
        }

    def predict(self, x, forward_feature=False):
        if forward_feature:
            return self.model.forward_features(x)
        prediction = self.model(x)
        if type(prediction) is tuple:
            return (prediction[0] + prediction[1]) / 2
        return prediction


class Fish(Algorithm):
    """
    Implementation of Fish, as seen in Gradient Matching for Domain
    Generalization, Shi et al. 2021.
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Fish, self).__init__(input_shape, num_classes, num_domains,
                                   hparams)
        self.input_shape = input_shape
        self.num_classes = num_classes

        self.network = networks.WholeFish(input_shape, num_classes, hparams)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.optimizer_inner_state = None

    def create_clone(self, device):
        self.network_inner = networks.WholeFish(self.input_shape, self.num_classes, self.hparams,
                                                weights=self.network.state_dict()).to(device)
        self.optimizer_inner = torch.optim.Adam(
            self.network_inner.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        if self.optimizer_inner_state is not None:
            self.optimizer_inner.load_state_dict(self.optimizer_inner_state)

    def fish(self, meta_weights, inner_weights, lr_meta):
        meta_weights = ParamDict(meta_weights)
        inner_weights = ParamDict(inner_weights)
        meta_weights += lr_meta * (inner_weights - meta_weights)
        return meta_weights

    def update(self, minibatches, unlabeled=None):
        self.create_clone(minibatches[0][0].device)

        for x, y in minibatches:
            loss = F.cross_entropy(self.network_inner(x), y)
            self.optimizer_inner.zero_grad()
            loss.backward()
            self.optimizer_inner.step()

        self.optimizer_inner_state = self.optimizer_inner.state_dict()
        meta_weights = self.fish(
            meta_weights=self.network.state_dict(),
            inner_weights=self.network_inner.state_dict(),
            lr_meta=self.hparams["meta_lr"]
        )
        self.network.reset_weights(meta_weights)

        return {'loss': loss.item()}

    def predict(self, x):
        return self.network(x)


class AbstractDANN(Algorithm):
    """Domain-Adversarial Neural Networks (abstract class)"""

    def __init__(self, input_shape, num_classes, num_domains,
                 hparams, conditional, class_balance):

        super(AbstractDANN, self).__init__(input_shape, num_classes, num_domains,
                                           hparams)

        self.register_buffer('update_count', torch.tensor([0]))
        self.conditional = conditional
        self.class_balance = class_balance

        # Algorithms
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])
        self.discriminator = networks.MLP(self.featurizer.n_outputs,
                                          num_domains, self.hparams)
        self.class_embeddings = nn.Embedding(num_classes,
                                             self.featurizer.n_outputs)

        # Optimizers
        self.disc_opt = torch.optim.Adam(
            (list(self.discriminator.parameters()) +
             list(self.class_embeddings.parameters())),
            lr=self.hparams["lr_d"],
            weight_decay=self.hparams['weight_decay_d'],
            betas=(self.hparams['beta1'], 0.9))

        self.gen_opt = torch.optim.Adam(
            (list(self.featurizer.parameters()) +
             list(self.classifier.parameters())),
            lr=self.hparams["lr_g"],
            weight_decay=self.hparams['weight_decay_g'],
            betas=(self.hparams['beta1'], 0.9))

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        self.update_count += 1
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        all_z = self.featurizer(all_x)
        if self.conditional:
            disc_input = all_z + self.class_embeddings(all_y)
        else:
            disc_input = all_z
        disc_out = self.discriminator(disc_input)
        disc_labels = torch.cat([
            torch.full((x.shape[0],), i, dtype=torch.int64, device=device)
            for i, (x, y) in enumerate(minibatches)
        ])

        if self.class_balance:
            y_counts = F.one_hot(all_y).sum(dim=0)
            weights = 1. / (y_counts[all_y] * y_counts.shape[0]).float()
            disc_loss = F.cross_entropy(disc_out, disc_labels, reduction='none')
            disc_loss = (weights * disc_loss).sum()
        else:
            disc_loss = F.cross_entropy(disc_out, disc_labels)

        disc_softmax = F.softmax(disc_out, dim=1)
        input_grad = autograd.grad(disc_softmax[:, disc_labels].sum(),
                                   [disc_input], create_graph=True)[0]
        grad_penalty = (input_grad ** 2).sum(dim=1).mean(dim=0)
        disc_loss += self.hparams['grad_penalty'] * grad_penalty

        d_steps_per_g = self.hparams['d_steps_per_g_step']
        if (self.update_count.item() % (1 + d_steps_per_g) < d_steps_per_g):

            self.disc_opt.zero_grad()
            disc_loss.backward()
            self.disc_opt.step()
            return {'disc_loss': disc_loss.item()}
        else:
            all_preds = self.classifier(all_z)
            classifier_loss = F.cross_entropy(all_preds, all_y)
            gen_loss = (classifier_loss +
                        (self.hparams['lambda'] * -disc_loss))
            self.disc_opt.zero_grad()
            self.gen_opt.zero_grad()
            gen_loss.backward()
            self.gen_opt.step()
            return {'gen_loss': gen_loss.item()}

    def predict(self, x):
        return self.classifier(self.featurizer(x))


class DANN(AbstractDANN):
    """Unconditional DANN"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(DANN, self).__init__(input_shape, num_classes, num_domains,
                                   hparams, conditional=False, class_balance=False)


#
#
# class CDANN(AbstractDANN):
#     """Conditional DANN"""
#
#     def __init__(self, input_shape, num_classes, num_domains, hparams):
#         super(CDANN, self).__init__(input_shape, num_classes, num_domains,
#                                     hparams, conditional=True, class_balance=True)
#
#
class IRM(ERM):
    """Invariant Risk Minimization"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(IRM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.register_buffer('update_count', torch.tensor([0]))

    @staticmethod
    def _irm_penalty(logits, y):
        device = "cuda" if logits[0][0].is_cuda else "cpu"
        scale = torch.tensor(1.).to(device).requires_grad_()
        loss_1 = F.cross_entropy(logits[::2] * scale, y[::2])
        loss_2 = F.cross_entropy(logits[1::2] * scale, y[1::2])
        grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
        result = torch.sum(grad_1 * grad_2)
        return result

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        penalty_weight = (self.hparams['irm_lambda'] if self.update_count
                                                        >= self.hparams['irm_penalty_anneal_iters'] else
                          1.0)
        nll = 0.
        penalty = 0.

        all_x = torch.cat([x for x, y in minibatches])
        all_logits = self.network(all_x)
        all_logits_idx = 0
        for i, (x, y) in enumerate(minibatches):
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll += F.cross_entropy(logits, y)
            penalty += self._irm_penalty(logits, y)
        nll /= len(minibatches)
        penalty /= len(minibatches)
        loss = nll + (penalty_weight * penalty)

        if self.update_count == self.hparams['irm_penalty_anneal_iters']:
            # Reset Adam, because it doesn't like the sharp jump in gradient
            # magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay'])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {'loss': loss.item(), 'nll': nll.item(),
                'penalty': penalty.item()}


class Fishr(Algorithm):
    "Invariant Gradients variances for Out-of-distribution Generalization"

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        assert backpack is not None, "Install backpack with: 'pip install backpack-for-pytorch==1.3.0'"
        super(Fishr, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.num_domains = num_domains

        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = extend(
            networks.Classifier(
                self.featurizer.n_outputs,
                num_classes,
                self.hparams['nonlinear_classifier'],
            )
        )
        self.network = nn.Sequential(self.featurizer, self.classifier)

        self.register_buffer("update_count", torch.tensor([0]))
        self.bce_extended = extend(nn.CrossEntropyLoss(reduction='none'))
        self.ema_per_domain = [
            MovingAverage(ema=self.hparams["ema"], oneminusema_correction=True)
            for _ in range(self.num_domains)
        ]
        self._init_optimizer()

    def _init_optimizer(self):
        self.optimizer = torch.optim.Adam(
            list(self.featurizer.parameters()) + list(self.classifier.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

    def update(self, minibatches, unlabeled=False):
        assert len(minibatches) == self.num_domains
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        len_minibatches = [x.shape[0] for x, y in minibatches]

        all_z = self.featurizer(all_x)
        all_logits = self.classifier(all_z)

        penalty = self.compute_fishr_penalty(all_logits, all_y, len_minibatches)
        all_nll = F.cross_entropy(all_logits, all_y)

        penalty_weight = 0
        if self.update_count >= self.hparams["penalty_anneal_iters"]:
            penalty_weight = self.hparams["lambda"]
            if self.update_count == self.hparams["penalty_anneal_iters"] != 0:
                # Reset Adam as in IRM or V-REx, because it may not like the sharp jump in
                # gradient magnitudes that happens at this step.
                self._init_optimizer()
        self.update_count += 1

        objective = all_nll + penalty_weight * penalty
        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {'loss': objective.item(), 'nll': all_nll.item(), 'penalty': penalty.item()}

    def compute_fishr_penalty(self, all_logits, all_y, len_minibatches):
        dict_grads = self._get_grads(all_logits, all_y)
        grads_var_per_domain = self._get_grads_var_per_domain(dict_grads, len_minibatches)
        return self._compute_distance_grads_var(grads_var_per_domain)

    def _get_grads(self, logits, y):
        self.optimizer.zero_grad()
        loss = self.bce_extended(logits, y).sum()
        with backpack(BatchGrad()):
            loss.backward(
                inputs=list(self.classifier.parameters()), retain_graph=True, create_graph=True
            )

        # compute individual grads for all samples across all domains simultaneously
        dict_grads = OrderedDict(
            [
                (name, weights.grad_batch.clone().view(weights.grad_batch.size(0), -1))
                for name, weights in self.classifier.named_parameters()
            ]
        )
        return dict_grads

    def _get_grads_var_per_domain(self, dict_grads, len_minibatches):
        # grads var per domain
        grads_var_per_domain = [{} for _ in range(self.num_domains)]
        for name, _grads in dict_grads.items():
            all_idx = 0
            for domain_id, bsize in enumerate(len_minibatches):
                env_grads = _grads[all_idx:all_idx + bsize]
                all_idx += bsize
                env_mean = env_grads.mean(dim=0, keepdim=True)
                env_grads_centered = env_grads - env_mean
                grads_var_per_domain[domain_id][name] = (env_grads_centered).pow(2).mean(dim=0)

        # moving average
        for domain_id in range(self.num_domains):
            grads_var_per_domain[domain_id] = self.ema_per_domain[domain_id].update(
                grads_var_per_domain[domain_id]
            )

        return grads_var_per_domain

    def _compute_distance_grads_var(self, grads_var_per_domain):

        # compute gradient variances averaged across domains
        grads_var = OrderedDict(
            [
                (
                    name,
                    torch.stack(
                        [
                            grads_var_per_domain[domain_id][name]
                            for domain_id in range(self.num_domains)
                        ],
                        dim=0
                    ).mean(dim=0)
                )
                for name in grads_var_per_domain[0].keys()
            ]
        )

        penalty = 0
        for domain_id in range(self.num_domains):
            penalty += l2_between_dicts(grads_var_per_domain[domain_id], grads_var)
        return penalty / self.num_domains

    def predict(self, x):
        return self.network(x)
