import gpflow
import numpy as np
from matplotlib import pyplot as plt

def auto_exclude(gp):

    mu, var = gp.predict_y(gp.data[0])
    to_remove = np.abs(mu - gp.data[1]) > 3 * np.sqrt(var)
    X_new = gp.data[0][~to_remove[:, 0]]
    Y_new = gp.data[1][~to_remove[:, 0]]

    kernel, likelihood = gp.kernel, gp.likelihood
    gp = gpflow.models.GPR(data=(X_new, Y_new),
                           kernel=kernel, likelihood=likelihood)

    print("Removed", np.sum(to_remove), "data points")

    return gp, to_remove

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

    def predict_y(self, X_star, N_star):

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
        y_star_var = prec_star**-1

        # Find generalised PoE GP predictive mean
        y_star_mean = np.zeros(N_star)
        for i in range(self.N_experts):
            y_star_mean += beta[:, i] * sigma_all[:, i]**-2 * mu_all[:, i]
        y_star_mean *= y_star_var

        return y_star_mean, y_star_var, beta
