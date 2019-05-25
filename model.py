from typing import *
from scipy.special import erf
import numpy as np
import torch
import torch.distributions as dist
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from kpi_anomaly_detection.evaluation_metric import ignore_missing
from network import MultiLinearGaussianStatistic
from network.loop import Loop, TestLoop
from torch_util import VstackDataset
from .donutx import CVAE, m_elbo, mcmc_missing_imputation, VAE, BasicVAE
from .kpi_frame_dataloader import KpiFrameDataLoader
from .kpi_frame_dataset import TimestampDataset, KpiFrameDataset
from .kpi_series import KPISeries
from .threshold_selection import threshold_map, threshold_prior


class DonutX:
    def __init__(self, max_epoch: int = 150, batch_size: int = 128, network_size: List[int] = None,
                 latent_dims: int = 8, window_size: int = 120, cuda: bool = True,
                 condition_dropout_left_rate=0.9, print_fn=print):
        if network_size is None:
            network_size = [100, 100]

        self.print_fn = print_fn
        self.condition_size = 60 + 24 + 7
        self.window_size = window_size
        self.latent_dims = latent_dims
        self.network_size = network_size
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.cuda = cuda
        self.condition_dropout_left_rate = condition_dropout_left_rate

        self._model = CVAE(
            MultiLinearGaussianStatistic(
                self.window_size + self.condition_size, self.latent_dims, self.network_size, eps=1e-4),
            MultiLinearGaussianStatistic(
                self.latent_dims + self.condition_size, self.window_size, self.network_size, eps=1e-4),
        )
        if self.cuda:
            self._model = self._model.cuda()

        if cuda:
            self.z_prior_dist = dist.Normal(
                Variable(torch.from_numpy(np.zeros((self.latent_dims,), np.float32)).cuda()),
                Variable(torch.from_numpy(np.ones((self.latent_dims,), np.float32)).cuda())
            )
        else:
            self.z_prior_dist = dist.Normal(
                Variable(torch.from_numpy(np.zeros((self.latent_dims,), np.float32))),
                Variable(torch.from_numpy(np.ones((self.latent_dims,), np.float32)))
            )

    def fit(self, kpi: KPISeries, valid_kpi: KPISeries = None):
        bernoulli = torch.distributions.Bernoulli(probs=self.condition_dropout_left_rate)
        self._model.train()
        with Loop(max_epochs=self.max_epoch, use_cuda=self.cuda, disp_epoch_freq=5,
                  print_fn=self.print_fn).with_context() as loop:
            optimizer = Adam(self._model.parameters(), lr=1e-3)
            lr_scheduler = StepLR(optimizer, step_size=10, gamma=0.75)
            train_timestamp_dataset = TimestampDataset(
                kpi, frame_size=self.window_size)
            train_kpiframe_dataset = KpiFrameDataset(kpi,
                                                     frame_size=self.window_size, missing_injection_rate=0.01)
            train_dataloader = KpiFrameDataLoader(VstackDataset(
                [train_kpiframe_dataset, train_timestamp_dataset]), batch_size=self.batch_size, shuffle=True,
                drop_last=True)
            if valid_kpi is not None:
                valid_timestamp_dataset = TimestampDataset(
                    valid_kpi.label_sampling(0.), frame_size=self.window_size)
                valid_kpiframe_dataset = KpiFrameDataset(valid_kpi,
                                                         frame_size=self.window_size, missing_injection_rate=0.)
                valid_dataloader = KpiFrameDataLoader(VstackDataset(
                    [valid_kpiframe_dataset, valid_timestamp_dataset]), batch_size=256, shuffle=True)
            else:
                valid_dataloader = None

            for epoch in loop.iter_epochs():
                lr_scheduler.step()
                for _, batch_data in loop.iter_steps(train_dataloader):
                    optimizer.zero_grad()
                    observe_x, observe_normal, observe_y = batch_data
                    if self.cuda:
                        mask = bernoulli.sample(sample_shape=observe_y.size()).cuda()
                    else:
                        mask = bernoulli.sample(sample_shape=observe_y.size())

                    observe_y = observe_y * mask
                    p_xz, q_zx, observe_z = self._model(
                        observe_x=observe_x, observe_y=observe_y)
                    loss = m_elbo(observe_x, observe_z, observe_normal, p_xz, q_zx,
                                  self.z_prior_dist) + self._model.penalty() * 0.001  # type: Variable
                    loss.backward()
                    clip_grad_norm_(self._model.parameters(), max_norm=10.)
                    optimizer.step()
                    loop.submit_metric("train_loss", loss.data)
                if valid_kpi is not None:
                    with torch.no_grad():
                        for _, batch_data in loop.iter_steps(valid_dataloader):
                            observe_x, observe_normal, observe_y = batch_data  # type: Variable, Variable
                            p_xz, q_zx, observe_z = self._model(
                                observe_x=observe_x, observe_y=observe_y)
                            loss = m_elbo(observe_x, observe_z, observe_normal, p_xz, q_zx,
                                          self.z_prior_dist) + self._model.penalty() * 0.001  # type: Variable
                            loop.submit_metric("valid_loss", loss.data)

    def predict(self, kpi: KPISeries, return_statistics=False, indicator_name="indicator"):
        """
        :param kpi:
        :param return_statistics:
        :param indicator_name:
            default "indicator": Reconstructed probability
            "indicator_prior": E_q(z|x)[log p(x|z) * p(z) / q(z|x)]
            "indicator_erf": erf(abs(x - x_mean) / x_std * scale_factor)
        :return:
        """
        with torch.no_grad():
            with TestLoop(use_cuda=True, print_fn=self.print_fn).with_context() as loop:
                test_timestamp_dataset = TimestampDataset(kpi, frame_size=self.window_size)
                test_kpiframe_dataset = KpiFrameDataset(kpi, frame_size=self.window_size, missing_injection_rate=0.0)
                test_dataloader = KpiFrameDataLoader(VstackDataset(
                    [test_kpiframe_dataset, test_timestamp_dataset]), batch_size=32, shuffle=False, drop_last=False)
                self._model.eval()
                for _, batch_data in loop.iter_steps(test_dataloader):
                    observe_x, observe_normal, observe_y = batch_data  # type: Variable, Variable
                    observe_x = mcmc_missing_imputation(observe_normal=observe_normal,
                                                        vae=self._model,
                                                        n_iteration=10,
                                                        observe_x=observe_x,
                                                        observe_y=observe_y
                                                        )
                    p_xz, q_zx, observe_z = self._model(observe_x=observe_x,
                                                        n_sample=128,
                                                        observe_y=observe_y)
                    loss = m_elbo(observe_x, observe_z, observe_normal,
                                  p_xz, q_zx, self.z_prior_dist)  # type: Variable
                    loop.submit_metric("test_loss", loss.data.cpu())

                    log_p_xz = p_xz.log_prob(observe_x).data.cpu().numpy()

                    log_p_x = log_p_xz * np.sum(
                        torch.exp(self.z_prior_dist.log_prob(observe_z) - q_zx.log_prob(observe_z)).cpu().numpy(),
                        axis=-1, keepdims=True)

                    indicator_erf = erf((torch.abs(observe_x - p_xz.mean) / p_xz.stddev).cpu().numpy() * 0.1589967)

                    loop.submit_data("indicator", -np.mean(log_p_xz[:, :, -1], axis=0))
                    loop.submit_data("indicator_prior", -np.mean(log_p_x[:, :, -1], axis=0))
                    loop.submit_data("indicator_erf", np.mean(indicator_erf[:, :, -1], axis=0))

                    loop.submit_data("x_mean", np.mean(
                        p_xz.mean.data.cpu().numpy()[:, :, -1], axis=0))
                    loop.submit_data("x_std", np.mean(
                        p_xz.stddev.data.cpu().numpy()[:, :, -1], axis=0))

            indicator = np.concatenate(loop.get_data_by_name(indicator_name))
            x_mean = np.concatenate(loop.get_data_by_name("x_mean"))
            x_std = np.concatenate(loop.get_data_by_name("x_std"))

            indicator = np.concatenate([np.ones(shape=self.window_size - 1) * np.min(indicator), indicator])
            if return_statistics:
                return indicator, x_mean, x_std
            else:
                return indicator

    def detect(self, kpi: KPISeries, train_kpi: KPISeries = None, return_threshold=False):
        indicators = self.predict(kpi)
        indicators_ignore_missing, *_ = ignore_missing(indicators, missing=kpi.missing)
        if train_kpi is not None:
            train_indicators = self.predict(train_kpi)
            train_indicators, train_labels = ignore_missing(train_indicators, train_kpi.label,
                                                            missing=train_kpi.missing)
            threshold = threshold_map(np.concatenate([indicators_ignore_missing, train_indicators]),
                                      np.concatenate(
                                          [np.ones_like(indicators_ignore_missing, dtype=np.int) * -1, train_labels]))
        else:
            threshold = threshold_prior(indicators_ignore_missing)

        predict = indicators >= threshold

        if return_threshold:
            return predict, threshold
        else:
            return predict


class Donut:
    def __init__(self, max_epoch: int = 150, batch_size: int = 128, network_size: List[int] = None,
                 latent_dims: int = 8, window_size: int = 120, cuda: bool = True, print_fn=print):
        if network_size is None:
            network_size = [100, 100]

        self.print_fn = print_fn
        self.window_size = window_size
        self.latent_dims = latent_dims
        self.network_size = network_size
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.cuda = cuda

        self._model = BasicVAE(
            MultiLinearGaussianStatistic(
                self.window_size, self.latent_dims, self.network_size, eps=1e-4),
            MultiLinearGaussianStatistic(
                self.latent_dims, self.window_size, self.network_size, eps=1e-4),
        )
        if self.cuda:
            self._model = self._model.cuda()

        if cuda:
            self.z_prior_dist = dist.Normal(
                Variable(torch.from_numpy(np.zeros((self.latent_dims,), np.float32)).cuda()),
                Variable(torch.from_numpy(np.ones((self.latent_dims,), np.float32)).cuda())
            )
        else:
            self.z_prior_dist = dist.Normal(
                Variable(torch.from_numpy(np.zeros((self.latent_dims,), np.float32))),
                Variable(torch.from_numpy(np.ones((self.latent_dims,), np.float32)))
            )

    def fit(self, kpi: KPISeries, valid_kpi: KPISeries = None):
        self._model.train()
        with Loop(max_epochs=self.max_epoch, use_cuda=self.cuda, disp_epoch_freq=5,
                  print_fn=self.print_fn).with_context() as loop:
            optimizer = Adam(self._model.parameters(), lr=1e-3)
            lr_scheduler = StepLR(optimizer, step_size=10, gamma=0.75)
            train_kpiframe_dataset = KpiFrameDataset(kpi,
                                                     frame_size=self.window_size, missing_injection_rate=0.01)
            train_dataloader = KpiFrameDataLoader(train_kpiframe_dataset, batch_size=self.batch_size, shuffle=True,
                                                  drop_last=True)
            if valid_kpi is not None:
                valid_kpiframe_dataset = KpiFrameDataset(valid_kpi,
                                                         frame_size=self.window_size, missing_injection_rate=0.)
                valid_dataloader = KpiFrameDataLoader(valid_kpiframe_dataset, batch_size=256, shuffle=True)
            else:
                valid_dataloader = None

            for epoch in loop.iter_epochs():
                lr_scheduler.step()
                for _, batch_data in loop.iter_steps(train_dataloader):
                    optimizer.zero_grad()
                    observe_x, observe_normal = batch_data

                    p_xz, q_zx, observe_z = self._model(observe_x=observe_x)
                    loss = m_elbo(observe_x, observe_z, observe_normal, p_xz, q_zx,
                                  self.z_prior_dist) + self._model.penalty() * 0.001  # type: Variable
                    loss.backward()
                    clip_grad_norm_(self._model.parameters(), max_norm=10.)
                    optimizer.step()
                    loop.submit_metric("train_loss", loss.data)
                if valid_kpi is not None:
                    with torch.no_grad():
                        for _, batch_data in loop.iter_steps(valid_dataloader):
                            observe_x, observe_normal = batch_data  # type: Variable, Variable
                            p_xz, q_zx, observe_z = self._model(observe_x=observe_x)
                            loss = m_elbo(observe_x, observe_z, observe_normal, p_xz, q_zx,
                                          self.z_prior_dist) + self._model.penalty() * 0.001  # type: Variable
                            loop.submit_metric("valid_loss", loss.data)
            # train_loss_epochs, train_loss = loop.get_metric_by_name("train_loss")
            # valid_loss_epochs, valid_loss = loop.get_metric_by_name("valid_loss")

    def predict(self, kpi: KPISeries, return_statistics=False, indicator_name="indicator"):
        """
        :param kpi:
        :param return_statistics:
        :param indicator_name:
            default "indicator": Reconstructed probability
            "indicator_prior": E_q(z|x)[log p(x|z) * p(z) / q(z|x)]
            "indicator_erf": erf(abs(x - x_mean) / x_std * scale_factor)
        :return:
        """
        with torch.no_grad():
            with TestLoop(use_cuda=True, print_fn=self.print_fn).with_context() as loop:
                test_kpiframe_dataset = KpiFrameDataset(kpi, frame_size=self.window_size, missing_injection_rate=0.0)
                test_dataloader = KpiFrameDataLoader(test_kpiframe_dataset, batch_size=32, shuffle=False,
                                                     drop_last=False)
                self._model.eval()
                for _, batch_data in loop.iter_steps(test_dataloader):
                    observe_x, observe_normal = batch_data  # type: Variable, Variable
                    observe_x = mcmc_missing_imputation(observe_normal=observe_normal,
                                                        vae=self._model,
                                                        n_iteration=10,
                                                        observe_x=observe_x,
                                                        )
                    p_xz, q_zx, observe_z = self._model(observe_x=observe_x,
                                                        n_sample=128,
                                                        )
                    loss = m_elbo(observe_x, observe_z, observe_normal,
                                  p_xz, q_zx, self.z_prior_dist)  # type: Variable
                    loop.submit_metric("test_loss", loss.data.cpu())

                    log_p_xz = p_xz.log_prob(observe_x).data.cpu().numpy()

                    log_p_x = log_p_xz * np.sum(
                        torch.exp(self.z_prior_dist.log_prob(observe_z) - q_zx.log_prob(observe_z)).cpu().numpy(),
                        axis=-1, keepdims=True)

                    indicator_erf = erf((torch.abs(observe_x - p_xz.mean) / p_xz.stddev).cpu().numpy() * 0.1589967)

                    loop.submit_data("indicator", -np.mean(log_p_xz[:, :, -1], axis=0))
                    loop.submit_data("indicator_prior", -np.mean(log_p_x[:, :, -1], axis=0))
                    loop.submit_data("indicator_erf", np.mean(indicator_erf[:, :, -1], axis=0))

                    loop.submit_data("x_mean", np.mean(
                        p_xz.mean.data.cpu().numpy()[:, :, -1], axis=0))
                    loop.submit_data("x_std", np.mean(
                        p_xz.stddev.data.cpu().numpy()[:, :, -1], axis=0))

            indicator = np.concatenate(loop.get_data_by_name(indicator_name))
            x_mean = np.concatenate(loop.get_data_by_name("x_mean"))
            x_std = np.concatenate(loop.get_data_by_name("x_std"))

            indicator = np.concatenate([np.ones(shape=self.window_size - 1) * np.min(indicator), indicator])
            if return_statistics:
                return indicator, x_mean, x_std
            else:
                return indicator

    def detect(self, kpi: KPISeries, train_kpi: KPISeries = None, return_threshold=False):
        indicators = self.predict(kpi)
        indicators_ignore_missing, *_ = ignore_missing(indicators, missing=kpi.missing)
        if train_kpi is not None:
            train_indicators = self.predict(train_kpi)
            train_indicators, train_labels = ignore_missing(train_indicators, train_kpi.label,
                                                            missing=train_kpi.missing)
            threshold = threshold_map(np.concatenate([indicators_ignore_missing, train_indicators]),
                                      np.concatenate(
                                          [np.ones_like(indicators_ignore_missing, dtype=np.int) * -1, train_labels]))
        else:
            threshold = threshold_prior(indicators_ignore_missing)

        predict = indicators >= threshold

        if return_threshold:
            return predict, threshold
        else:
            return predict
