# Copyright 2016-2020 The GPflow Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

import tensorflow as tf
from check_shapes import check_shapes, inherit_check_shapes

import gpflow

from .. import posteriors
from ..base import InputData, MeanAndVariance, RegressionData, TensorData
from ..kernels import Kernel
from ..likelihoods import Gaussian
from ..logdensities import multivariate_normal
from ..mean_functions import MeanFunction
from ..utilities import add_likelihood_noise_cov, assert_params_false
from .model import GPModel
from .training_mixins import InternalDataTrainingLossMixin
from .util import data_input_to_tensor


class GPR_deprecated(GPModel, InternalDataTrainingLossMixin):
    r"""
    Gaussian Process Regression.

    This is a vanilla implementation of GP regression with a Gaussian
    likelihood.  Multiple columns of Y are treated independently.

    The log likelihood of this model is given by

    .. math::
       \log p(Y \,|\, \mathbf f) =
            \mathcal N(Y \,|\, 0, \sigma_n^2 \mathbf{I})

    To train the model, we maximise the log _marginal_ likelihood
    w.r.t. the likelihood variance and kernel hyperparameters theta.
    The marginal likelihood is found by integrating the likelihood
    over the prior, and has the form

    .. math::
       \log p(Y \,|\, \sigma_n, \theta) =
            \mathcal N(Y \,|\, 0, \mathbf{K} + \sigma_n^2 \mathbf{I})

    For a use example see :doc:`../../../../notebooks/getting_started/basic_usage`.
    """

    @check_shapes(
        "data[0]: [N, D]",
        "data[1]: [N, P]",
        "noise_variance: []",
    )
    def __init__(
        self,
        data: RegressionData,
        kernel: Kernel,
        mean_function: Optional[MeanFunction] = None,
        noise_variance: Optional[TensorData] = None,
        likelihood: Optional[Gaussian] = None,
    ):
        assert (noise_variance is None) or (
            likelihood is None
        ), "Cannot set both `noise_variance` and `likelihood`."
        if likelihood is None:
            if noise_variance is None:
                noise_variance = 1.0
            likelihood = gpflow.likelihoods.Gaussian(noise_variance)
        _, Y_data = data
        super().__init__(kernel, likelihood, mean_function, num_latent_gps=Y_data.shape[-1])
        self.data = data_input_to_tensor(data)

    # type-ignore is because of changed method signature:
    @inherit_check_shapes
    def maximum_log_likelihood_objective(self) -> tf.Tensor:  # type: ignore[override]
        return self.log_marginal_likelihood()

    @check_shapes(
        "return: []",
    )
    def log_marginal_likelihood(self) -> tf.Tensor:
        r"""
        Computes the log marginal likelihood.

        .. math::
            \log p(Y | \theta).

        """
        X, Y = self.data
        K = self.kernel(X)
        ks = add_likelihood_noise_cov(K, self.likelihood, X)
        L = tf.linalg.cholesky(ks)
        m = self.mean_function(X)

        # [R,] log-likelihoods for each independent dimension of Y
        log_prob = multivariate_normal(Y, m, L)
        return tf.reduce_sum(log_prob)

    @inherit_check_shapes
    def predict_f(
        self, Xnew: InputData, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        r"""
        This method computes predictions at X \in R^{N \x D} input points

        .. math::
            p(F* | Y)

        where F* are points on the GP at new data points, Y are noisy observations at training data
        points.
        """
        assert_params_false(self.predict_f, full_output_cov=full_output_cov)

        X, Y = self.data
        err = Y - self.mean_function(X)

        kmm = self.kernel(X)
        knn = self.kernel(Xnew, full_cov=full_cov)
        kmn = self.kernel(X, Xnew)
        kmm_plus_s = add_likelihood_noise_cov(kmm, self.likelihood, X)

        conditional = gpflow.conditionals.base_conditional
        f_mean_zero, f_var = conditional(
            kmn, kmm_plus_s, knn, err, full_cov=full_cov, white=False
        )  # [N, P], [N, P] or [P, N, N]
        f_mean = f_mean_zero + self.mean_function(Xnew)
        return f_mean, f_var


class GPR_with_posterior(GPR_deprecated):
    """
    This is an implementation of GPR that provides a posterior() method that
    enables caching for faster subsequent predictions.
    """

    def posterior(
        self,
        precompute_cache: posteriors.PrecomputeCacheType = posteriors.PrecomputeCacheType.TENSOR,
    ) -> posteriors.GPRPosterior:
        """
        Create the Posterior object which contains precomputed matrices for
        faster prediction.

        precompute_cache has three settings:

        - `PrecomputeCacheType.TENSOR` (or `"tensor"`): Precomputes the cached
          quantities and stores them as tensors (which allows differentiating
          through the prediction). This is the default.
        - `PrecomputeCacheType.VARIABLE` (or `"variable"`): Precomputes the cached
          quantities and stores them as variables, which allows for updating
          their values without changing the compute graph (relevant for AOT
          compilation).
        - `PrecomputeCacheType.NOCACHE` (or `"nocache"` or `None`): Avoids
          immediate cache computation. This is useful for avoiding extraneous
          computations when you only want to call the posterior's
          `fused_predict_f` method.
        """

        return posteriors.GPRPosterior(
            kernel=self.kernel,
            data=self.data,
            likelihood=self.likelihood,
            mean_function=self.mean_function,
            precompute_cache=precompute_cache,
        )

    @inherit_check_shapes
    def predict_f(
        self, Xnew: InputData, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        """
        For backwards compatibility, GPR's predict_f uses the fused (no-cache)
        computation, which is more efficient during training.

        For faster (cached) prediction, predict directly from the posterior object, i.e.,:
            model.posterior().predict_f(Xnew, ...)
        """
        return self.posterior(posteriors.PrecomputeCacheType.NOCACHE).fused_predict_f(
            Xnew, full_cov=full_cov, full_output_cov=full_output_cov
        )


class GPR(GPR_with_posterior):
    # subclassed to ensure __class__ == "GPR"

    __doc__ = GPR_deprecated.__doc__  # Use documentation from GPR_deprecated.


class PoE_GP():

    def __init__(self):
        self.N_experts = 0
        self.experts = []

    def add_gp(self, gp):
        self.experts.append(gp)
        self.N_experts += 1

    def training_loss(self):
        loss = 0
        for i in range(self.N_experts):
            loss += -self.experts[i].log_marginal_likelihood()
        return loss

    def predict(self, X_star, N_star):

        # Run all experts, collection their predictive means and
        # standard deviation
        mu_all = np.zeros([N_star, self.N_experts])
        sigma_all = np.zeros([N_star, self.N_experts])
        for i in range(self.N_experts):
            mu, var = self.experts[i].predict_y(X_star[:, None])
            mu_all[:, i] = mu.numpy().flatten()
            sigma_all[:, i] = np.sqrt(var.numpy()).flatten()

        # Calculate the normalised predictive power of the predictions made
        # by each GP. Note that here we are assuming that k(x_star, x_star)=1
        # to simplify the calculation. Note that we also add a small 'jitter'
        # term to the prior, to ensure that beta never goes below zero.
        beta = np.zeros([N_star, self.N_experts])
        for i in range(self.N_experts):
            noise_std = np.sqrt(self.experts[i].likelihood.variance.numpy())
            beta[:, i] = (0.5 * np.log(1 + 1e-9 + noise_std**2) -
                            np.log(sigma_all[:, i]))

        # Normalise beta
        for i in range(N_star):
            beta[i, :] = beta[i, :] / np.sum(beta[i, :])

        # Find generalised PoE GP predictive precision
        prec_star = np.zeros(N_star)
        for i in range(self.N_experts):
            prec_star += beta[:, i] * sigma_all[:, i]**-2

        # Find generalised PoE GP predictive variance and standard
        # deviation
        var_star = prec_star**-1
        y_star_std = var_star**0.5

        # Find generalised PoE GP predictive mean
        y_star_mean = np.zeros(N_star)
        for i in range(self.N_experts):
            y_star_mean += beta[:, i] * sigma_all[:, i]**-2 * mu_all[:, i]
        y_star_mean *= var_star

        return y_star_mean, y_star_std, beta

    def auto_exclude(self, plots=False):
        for i in range(self.N_experts):
            mu, var = self.experts[i].predict_y(self.experts[i].data[0])
            to_remove = np.abs(mu.numpy() - self.experts[i].data[1]) > 3 * np.sqrt(var.numpy())
            X_new = self.experts[i].data[0][~to_remove]
            Y_new = self.experts[i].data[1][~to_remove]

            updated_expert = gpflow.models.GPR(data=(X_new[:, None], Y_new[:, None]), kernel=self.experts[i].kernel, likelihood=self.experts[i].likelihood)

            if plots:
                fig, ax = plt.subplots()
                ax.plot(self.experts[i].data[0], self.experts[i].data[1], 'o')
                ax.plot(self.experts[i].data[0][to_remove],
                        self.experts[i].data[1][to_remove], 'o')
                ax.plot(self.experts[i].data[0], mu, 'black')
                ax.plot(self.experts[i].data[0], mu + 3 * np.sqrt(var), 'black')
                ax.plot(self.experts[i].data[0], mu - 3 * np.sqrt(var), 'black')

            self.experts[i] = updated_expert
