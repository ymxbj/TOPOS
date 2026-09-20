"""Euler sampler for flow-matching models (with optional classifier-free guidance)."""
from typing import Any, Optional

import numpy as np
import torch
from easydict import EasyDict as edict
from tqdm import tqdm

from .base import Sampler


class FlowEulerSampler(Sampler):
    def __init__(self, sigma_min: float):
        self.sigma_min = sigma_min

    def _v_to_xstart_eps(self, x_t, t, v):
        eps = (1 - t) * v + x_t
        x_0 = (1 - self.sigma_min) * x_t - (self.sigma_min + (1 - self.sigma_min) * t) * v
        return x_0, eps

    def _inference_model(self, model, x_t, t, cond=None, **kwargs):
        t = torch.tensor([1000 * t] * x_t.shape[0], device=x_t.device, dtype=torch.float32)
        if cond is not None and cond.shape[0] == 1 and x_t.shape[0] > 1:
            cond = cond.repeat(x_t.shape[0], *([1] * (len(cond.shape) - 1)))
        return model(x_t, t, cond, **kwargs)

    def _get_model_prediction(self, model, x_t, t, cond=None, **kwargs):
        pred_v = self._inference_model(model, x_t, t, cond, **kwargs)
        pred_x_0, pred_eps = self._v_to_xstart_eps(x_t=x_t, t=t, v=pred_v)
        return pred_x_0, pred_eps, pred_v

    @torch.no_grad()
    def sample_once(self, model, x_t, t: float, t_prev: float, cond: Optional[Any] = None, **kwargs):
        pred_x_0, _, pred_v = self._get_model_prediction(model, x_t, t, cond, **kwargs)
        pred_x_prev = x_t - (t - t_prev) * pred_v
        return edict({'pred_x_prev': pred_x_prev, 'pred_x_0': pred_x_0})

    @torch.no_grad()
    def sample(self, model, noise, cond: Optional[Any] = None, steps: int = 50,
               rescale_t: float = 1.0, verbose: bool = True, **kwargs):
        sample = noise
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        pairs = list(zip(t_seq[:-1], t_seq[1:]))

        ret = edict({'samples': None, 'pred_x_t': [], 'pred_x_0': []})
        for t, t_prev in tqdm(pairs, desc='Sampling', disable=not verbose):
            out = self.sample_once(model, sample, t, t_prev, cond, **kwargs)
            sample = out.pred_x_prev
            ret.pred_x_t.append(out.pred_x_prev)
            ret.pred_x_0.append(out.pred_x_0)
        ret.samples = sample
        return ret


class _CFGMixin:
    def _inference_model(self, model, x_t, t, cond, neg_cond, cfg_strength, **kwargs):
        pred = super()._inference_model(model, x_t, t, cond, **kwargs)
        neg_pred = super()._inference_model(model, x_t, t, neg_cond, **kwargs)
        return (1 + cfg_strength) * pred - cfg_strength * neg_pred


class FlowEulerCfgSampler(_CFGMixin, FlowEulerSampler):
    """Flow-matching Euler sampler with classifier-free guidance."""

    @torch.no_grad()
    def sample(self, model, noise, cond, neg_cond, steps: int = 50, rescale_t: float = 1.0,
               cfg_strength: float = 3.0, verbose: bool = True, **kwargs):
        return super().sample(model, noise, cond, steps, rescale_t, verbose,
                              neg_cond=neg_cond, cfg_strength=cfg_strength, **kwargs)
