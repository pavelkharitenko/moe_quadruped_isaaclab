import torch, pyro
import pyro.distributions as dist
from pyro.infer import SVI, TraceMeanField_ELBO, TraceEnum_ELBO
from pyro.optim import ClippedAdam
import numpy as np

from itertools import cycle
import numpy as np


def kl_diag_gaussian(mu1, var1, mu2, var2):
    """
    KL( N(mu1, var1) || N(mu2, var2) )
    """
    return 0.5 * torch.sum(
        torch.log(var2 / var1)
        + (var1 + (mu1 - mu2) ** 2) / var2
        - 1.0
    )


class PyroDPMM:
    """
    Truncated stick-breaking DPMM
    """
    def __init__(self, latent_dim, K, gamma0, device):
        self.latent_dim = latent_dim
        self.K = K
        self.gamma0 = gamma0
        self.device = device

    def model(self, z):
        N, D = z.shape

        with pyro.plate("components", self.K):
            beta = pyro.sample("beta", dist.Beta(1.0, self.gamma0))
            mu = pyro.sample(
                "mu",
                dist.Normal(0, 5).expand([D]).to_event(1)
            )
            sigma = pyro.sample(
                "sigma",
                dist.LogNormal(0.0, 1.0).expand([D]).to_event(1)
            )

        # Stick-breaking weights
        stick = beta
        weights = stick * torch.cumprod(
            torch.cat(
                [torch.ones(1, device=self.device), 1 - stick[:-1]]
            ),
            dim=0,
        )

        with pyro.plate("data", N):
            assignment = pyro.sample(
                "assignment", dist.Categorical(weights),
                infer={"enumerate": "parallel"}
            )
            pyro.sample(
                "obs",
                dist.Normal(mu[assignment], sigma[assignment]).to_event(1),
                obs=z,
            )

    def guide(self, z):
        K, D = self.K, self.latent_dim

        beta_q = pyro.param(
            "beta_q",
            torch.ones(K, 2, device=self.device),
            constraint=dist.constraints.positive,
        )

        # currently with mean-field assumption, pointwise instead of NIW
        mu_q = pyro.param(
            "mu_q",
            torch.zeros(K, D, device=self.device),
        )
        sigma_q = pyro.param(
            "sigma_q",
            torch.ones(K, D, device=self.device),
            constraint=dist.constraints.positive,
        )

        with pyro.plate("components", K):
            pyro.sample("beta", dist.Beta(beta_q[:, 0], beta_q[:, 1]))
            pyro.sample("mu", dist.Normal(mu_q, 1.0).to_event(1))
            pyro.sample("sigma", dist.LogNormal(torch.log(sigma_q), 0.1).to_event(1))

class PyroBNPModel:
    """
    Drop-in replacement for bnpy-based BNPModel.
    Designed for DPMM-VAE.
    """

    def __init__(
        self,
        latent_dim,
        gamma0=5.0,
        K_init=1,
        device=None,
        lr=1e-3,
    ):
        self.latent_dim = latent_dim
        self.gamma0 = gamma0
        self.K = K_init

        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        # Latent buffer Z (Algorithm 1, step 5)
        self.Z_buffer = []

        self.dpmm = PyroDPMM(latent_dim, self.K, gamma0, self.device)
        self.optimizer = ClippedAdam({"lr": lr})
        self.svi = SVI(
            self.dpmm.model,
            self.dpmm.guide,
            self.optimizer,
            loss=TraceMeanField_ELBO(),
        )
        self.svi = SVI(
            self.dpmm.model,
            self.dpmm.guide,
            self.optimizer,
            loss=TraceEnum_ELBO(max_plate_nesting=2),
        )

        self.comp_mu = None
        self.comp_var = None

    def cluster_assignments(self, z):
        """
        Equivalent to bnpy.calc_local_params
        """
        z = z.to(self.device)

        mu = self.comp_mu.to(self.device)
        var = self.comp_var.to(self.device)

        log_probs = []
        for k in range(self.K):
            dist_k = torch.distributions.Normal(mu[k], var[k].sqrt())
            log_probs.append(dist_k.log_prob(z).sum(dim=1))

        log_probs = torch.stack(log_probs, dim=1)
        resp = torch.softmax(log_probs, dim=1)
        Z = resp.argmax(dim=1)

        return resp.detach().cpu(), Z.detach().cpu()

    def fit(self, z, num_steps=200):
        """
        Equivalent to bnpy.run(...)
        """
        z = z.detach().to(self.device)

        pyro.clear_param_store()
        for _ in range(num_steps):
            self.svi.step(z)

        self._update_component_params()


    def _update_component_params(self):
        self.comp_mu = pyro.param("mu_q").detach().cpu()
        self.comp_var = pyro.param("sigma_q").pow(2).detach().cpu()


    def birth_move(self, min_weight=50, var_thresh=1.0):
        """
        Heuristic birth (split high-variance clusters)
        """
        Nk = self._component_usage()

        new_mu, new_var = [], []

        for k in range(self.K):
            mu, var = self.comp_mu[k], self.comp_var[k]

            if Nk[k] > min_weight and var.mean() > var_thresh:
                eps = 0.1 * torch.randn_like(mu)
                new_mu += [mu + eps, mu - eps]
                new_var += [var.clone(), var.clone()]
            else:
                new_mu.append(mu)
                new_var.append(var)

        self._reset_components(new_mu, new_var)


    def merge_move(self, kl_thresh=0.1, min_weight=10):
        Nk = self._component_usage()

        merged = set()
        new_mu, new_var = [], []

        for i in range(self.K):
            if i in merged or Nk[i] < min_weight:
                continue

            for j in range(i + 1, self.K):
                if j in merged or Nk[j] < min_weight:
                    continue

                kl = kl_diag_gaussian(
                    self.comp_mu[i], self.comp_var[i],
                    self.comp_mu[j], self.comp_var[j],
                )

                if kl < kl_thresh:
                    wi, wj = Nk[i], Nk[j]
                    mu = (wi * self.comp_mu[i] + wj * self.comp_mu[j]) / (wi + wj)
                    var = (wi * self.comp_var[i] + wj * self.comp_var[j]) / (wi + wj)

                    new_mu.append(mu)
                    new_var.append(var)
                    merged.update([i, j])
                    break

            if i not in merged:
                new_mu.append(self.comp_mu[i])
                new_var.append(self.comp_var[i])

        self._reset_components(new_mu, new_var)


    def _reset_components(self, new_mu, new_var):
        self.K = len(new_mu)
        self.dpmm.K = self.K

        pyro.clear_param_store()

        pyro.param("mu_q", torch.stack(new_mu).to(self.device))
        pyro.param("sigma_q", torch.sqrt(torch.stack(new_var)).to(self.device))
        pyro.param(
            "beta_q",
            torch.ones(self.K, 2, device=self.device),
            constraint=dist.constraints.positive,
        )


    def _component_usage(self):
        Z = torch.cat(self.Z_buffer, dim=0)
        resp, _ = self.cluster_assignments(Z)
        return resp.sum(dim=0)

    def sample_component(self, num_samples, k):
        mu, var = self.comp_mu[k], self.comp_var[k]
        return torch.distributions.Normal(mu, var.sqrt()).sample((num_samples,))

    def sample_all(self, num_samples):
        num_per = num_samples // self.K
        return torch.cat(
            [self.sample_component(num_per, k) for k in range(self.K)],
            dim=0,
        )
