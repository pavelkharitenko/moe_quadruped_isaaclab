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

    def fit(
        self,
        z,
        num_laps=20,
        svi_steps_per_lap=10,
    ):
        z = z.detach().to(self.device)

        for lap in range(num_laps):
            # Variational updates
            for _ in range(svi_steps_per_lap):
                self.svi.step(z)

            self._update_component_params()
            self.Z_buffer.append(z.detach().cpu())

            # Birth
            if lap >= 1:
                self.birth_move_elbo(z)

            # Merge
            if lap >= 2:
                self.merge_move_elbo(z)



    def _update_component_params(self):
        self.comp_mu = pyro.param("mu_q").detach().cpu()
        self.comp_var = pyro.param("sigma_q").pow(2).detach().cpu()


   
    def __reset_components(self, new_mu, new_var):
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

    def _reset_components(self, new_mu, new_var):
        # ensure list of tensors
        if isinstance(new_mu, torch.Tensor):
            new_mu = [new_mu]
        if isinstance(new_var, torch.Tensor):
            new_var = [new_var]

        # Flatten each tensor to 1D so stacking works correctly
        new_mu = [mu.view(-1) if mu.dim() > 1 else mu for mu in new_mu]
        new_var = [var.view(-1) if var.dim() > 1 else var for var in new_var]

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
        if len(self.Z_buffer) == 0:
            return torch.zeros(self.K)
        
        Z = torch.cat(self.Z_buffer, dim=0).to(self.device)
        resp, _ = self.cluster_assignments(Z)  # resp.shape = [N, current K]
        
        # ensure we always return length self.K
        usage = torch.zeros(self.K, device=self.device)
        K_curr = resp.shape[1]
        usage[:K_curr] = resp.sum(dim=0)
        return usage


    def __component_usage(self):
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


    def estimate_elbo(self, z, num_particles=5):
        elbo = 0.0
        for _ in range(num_particles):
            elbo += self.svi.loss(self.dpmm.model, self.dpmm.guide, z)
        return -elbo / num_particles  # higher is better
    

    def poorly_explained_points(self, z, frac=0.2):
        resp, _ = self.cluster_assignments(z)
        max_resp = resp.max(dim=1).values
        thresh = torch.quantile(max_resp, frac)
        return z[max_resp < thresh]

    def birth_move_elbo(
        self,
        z,
        frac_poor=0.2,
        min_points=0,
        elbo_tol=1e-3,
    ):
        """
        ELBO-based birth move (memoVB-style)
        """
        z = z.to(self.device)

        # Step 1: find poorly explained points
        z_bad = self.poorly_explained_points(z, frac=frac_poor)
        if z_bad.shape[0] < min_points:
            return False

        # Step 2: baseline ELBO
        elbo_before = self.estimate_elbo(z)

        # Step 3: propose split via simple 2-means
        mu_old = self.comp_mu.clone()
        var_old = self.comp_var.clone()

        k = torch.argmax(self._component_usage())
        mu = mu_old[k]
        eps = 0.1 * torch.randn_like(mu)

        new_mu = list(mu_old)
        new_var = list(var_old)

        new_mu[k] = mu + eps
        new_mu.append(mu - eps)
        new_var.append(var_old[k].clone())

        # Step 4: reset model with proposed structure
        self._reset_components(new_mu, new_var)

        # Re-optimize briefly
        for _ in range(20):
            self.svi.step(z)

        self._update_component_params()

        # Step 5: ELBO comparison
        elbo_after = self.estimate_elbo(z)

        if elbo_after > elbo_before + elbo_tol:
            print(f"[BIRTH ACCEPTED] ELBO {elbo_before:.2f} → {elbo_after:.2f}")
            return True
        else:
            # revert
            self._reset_components(mu_old, var_old)
            print(f"[BIRTH REJECTED]")
            return False

    def merge_move_elbo(
        self,
        z,
        kl_thresh=0.5,
        elbo_tol=1e-3,
    ):
        z = z.to(self.device)
        elbo_before = self.estimate_elbo(z)

        for i in range(self.K):
            for j in range(i + 1, self.K):
                kl = kl_diag_gaussian(
                    self.comp_mu[i], self.comp_var[i],
                    self.comp_mu[j], self.comp_var[j],
                )
                if kl > kl_thresh:
                    continue

                # Propose merge
                mu_old = self.comp_mu.clone()
                var_old = self.comp_var.clone()

                wi, wj = self._component_usage()[[i, j]]
                mu = (wi * mu_old[i] + wj * mu_old[j]) / (wi + wj)
                var = (wi * var_old[i] + wj * var_old[j]) / (wi + wj)

                new_mu = [mu_old[k] for k in range(self.K) if k not in (i, j)]
                new_var = [var_old[k] for k in range(self.K) if k not in (i, j)]
                new_mu.append(mu)
                new_var.append(var)

                self._reset_components(new_mu, new_var)

                for _ in range(20):
                    self.svi.step(z)

                self._update_component_params()
                elbo_after = self.estimate_elbo(z)

                if elbo_after > elbo_before + elbo_tol:
                    print(f"[MERGE ACCEPTED] {i},{j}")
                    return True
                else:
                    self._reset_components(mu_old, var_old)

        return False
