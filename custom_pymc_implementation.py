import theano.tensor as tt

import pymc3 as pm
from pymc3.gp.cov import Constant
from pymc3.gp.mean import Zero
from pymc3.gp.util import (conditioned_vars,
                           infer_shape, stabilize, cholesky
                           )

from pymc3.gp import Latent


@conditioned_vars(["X", "f"])
class CustomLatent(Latent):
    R"""
    Modified Latent Gaussian process of Latent from pymc3
    """

    def __init__(self, mean_func=Zero(), cov_func=Constant(0.0)):
        super(CustomLatent, self).__init__(mean_func, cov_func)

    def _build_prior(self, name, X, reparameterize=True, **kwargs):
        mu = self.mean_func(X.flatten())
        chol = cholesky(stabilize(self.cov_func(X)))
        shape = infer_shape(X, kwargs.pop("shape", None))
        if reparameterize:
            v = pm.Normal(name + "_rotated_", mu=0.0, sd=1.0, shape=shape * 2, **kwargs)
            f = pm.Deterministic(name, mu + tt.dot(chol, tt.reshape(v, (shape, 2))).flatten())
        else:
            f = pm.MvNormal(name, mu=mu, chol=chol, shape=shape, **kwargs)
        return f