"""
Transport module for SCLDM models.

Provides functionality for creating transport objects, defining model, path, and weight types,
and handling sampling and prior log probability computations.
"""

# pyright: reportUnknownMemberType=false

import enum
from collections.abc import Callable
from typing import Any, cast

import numpy as np
import torch as th
from torchdiffeq import odeint  # type: ignore[import]

from .sde_path import GVPCPlan, ICPlan, VPCPlan, expand_t_like_x


def mean_flat(x: th.Tensor) -> th.Tensor:
    """Take the mean over all non-batch dimensions."""
    return th.mean(x, dim=list(range(1, len(x.size()))))


class sde:
    """SDE solver class."""

    def __init__(
        self,
        drift: Callable[..., Any],
        diffusion: Callable[..., Any],
        *,
        t0: float,
        t1: float,
        num_steps: int,
        sampler_type: str,
    ):
        """Initialize the SDE solver with the given drift and diffusion functions, time interval, number of steps, and sampler type."""
        assert t0 < t1, "SDE sampler has to be in forward time"

        self.num_timesteps = num_steps
        self.t = th.linspace(t0, t1, num_steps)
        self.dt = self.t[1] - self.t[0]
        self.drift = drift
        self.diffusion = diffusion
        self.sampler_type = sampler_type

    def __Euler_Maruyama_step(
        self,
        x: th.Tensor,
        mean_x: th.Tensor,
        t: th.Tensor,
        model: Callable[..., Any],
        **model_kwargs: dict[str, Any],
    ) -> tuple[th.Tensor, th.Tensor]:
        w_cur = th.randn(x.size()).to(x)
        t = th.ones(x.size(0)).to(x) * t
        dw = w_cur * th.sqrt(self.dt)
        drift = self.drift(x, t, model, **model_kwargs)
        diffusion = self.diffusion(x, t)
        mean_x = x + drift * self.dt
        x = mean_x + th.sqrt(2 * diffusion) * dw
        return x, mean_x

    def __Heun_step(
        self,
        x: th.Tensor,
        _: th.Tensor,
        t: th.Tensor,
        model: Callable[..., Any],
        **model_kwargs: dict[str, Any],
    ) -> tuple[th.Tensor, th.Tensor]:
        w_cur = th.randn(x.size()).to(x)
        dw = w_cur * th.sqrt(self.dt)
        t_cur = th.ones(x.size(0)).to(x) * t
        diffusion = self.diffusion(x, t_cur)
        xhat = x + th.sqrt(2 * diffusion) * dw
        K1 = self.drift(xhat, t_cur, model, **model_kwargs)
        xp = xhat + self.dt * K1
        K2 = self.drift(xp, t_cur + self.dt, model, **model_kwargs)
        return (
            xhat + 0.5 * self.dt * (K1 + K2),
            xhat,
        )  # at last time point we do not perform the heun step

    def __forward_fn(self):
        """TODO: generalize here by adding all private functions ending with steps to it."""
        sampler_dict = {
            "Euler": self.__Euler_Maruyama_step,
            "Heun": self.__Heun_step,
        }

        try:
            sampler = sampler_dict[self.sampler_type]
        except KeyError as e:
            raise NotImplementedError("Sampler type not implemented.") from e

        return sampler

    def sample(
        self, init: th.Tensor, model: Callable[..., Any], **model_kwargs: dict[str, Any]
    ) -> list[th.Tensor]:
        """
        Forward loop of sde sampling.

        Args:
        ----
            init: initial state tensor
            model: model to use for computing the drift
            model_kwargs: additional keyword arguments for the model

        Returns:
        -------
            samples: list of sampled states at each time step
        """
        x = init
        mean_x = init
        samples: list[th.Tensor] = []
        sampler = self.__forward_fn()
        for ti in self.t[:-1]:
            with th.no_grad():
                x, mean_x = sampler(x, mean_x, ti, model, **model_kwargs)
                samples.append(x)

        return samples


class ode:
    """ODE solver class."""

    def __init__(
        self,
        drift: Callable[..., Any],
        *,
        t0: float,
        t1: float,
        sampler_type: str,
        num_steps: int,
        atol: float,
        rtol: float,
    ):
        """Initialize the ODE solver with the given drift function, time interval, sampler type, number of steps, and tolerances."""
        assert t0 < t1, "ODE sampler has to be in forward time"

        self.drift = drift
        self.t = th.linspace(t0, t1, num_steps)
        self.atol = atol
        self.rtol = rtol
        self.sampler_type = sampler_type

    def sample(
        self,
        x: th.Tensor | tuple[th.Tensor, ...],
        model: Callable[..., Any],
        **model_kwargs: dict[str, Any],
    ) -> th.Tensor:
        """
        Sample from the ODE solver using the given model and input.

        Args:
        ----
            x: input tensor or tuple of tensors
            model: model to use for computing the drift
            model_kwargs: additional keyword arguments for the model

        Returns:
        -------
            samples: sampled trajectory from the ODE solver
        """
        device = x[0].device if isinstance(x, tuple) else x.device

        def _fn(t: th.Tensor, x: th.Tensor) -> th.Tensor:
            t = (
                th.ones(x[0].size(0)).to(device) * t
                if isinstance(x, tuple)
                else th.ones(x.size(0)).to(device) * t
            )
            model_output = self.drift(x, t, model, **model_kwargs)
            return model_output

        t = self.t.to(device)
        atol = cast(float, [self.atol] * len(x) if isinstance(x, tuple) else [self.atol])
        rtol = cast(float, [self.rtol] * len(x) if isinstance(x, tuple) else [self.rtol])
        samples = cast(
            th.Tensor,
            odeint(_fn, x, t, method=self.sampler_type, atol=atol, rtol=rtol),
        )
        return samples


class WeightType(enum.Enum):
    """Which type of weighting to use."""

    NONE = enum.auto()
    VELOCITY = enum.auto()
    LIKELIHOOD = enum.auto()


class ModelType(enum.Enum):
    """Which type of output the model predicts."""

    NOISE = enum.auto()  # the model predicts epsilon
    SCORE = enum.auto()  # the model predicts \nabla \log p(x)
    VELOCITY = enum.auto()  # the model predicts v(x)


class PathType(enum.Enum):
    """Which type of path to use."""

    LINEAR = enum.auto()
    GVP = enum.auto()
    VP = enum.auto()


def create_transport(
    path_type: str = "Linear",
    prediction: str = "velocity",
    loss_weight: str | None = None,
    train_eps: float | None = None,
    sample_eps: float | None = None,
):
    """
    Function for creating Transport object.

    **Note**: model prediction defaults to velocity

    Args:
    ----
        path_type: type of path to use; default to linear
        prediction: model prediction target; one of "velocity", "score", or "noise"
        loss_weight: loss weighting scheme; one of "velocity", "likelihood", or None
        train_eps: small epsilon for avoiding instability during training
        sample_eps: small epsilon for avoiding instability during sampling
    """
    if prediction == "noise":
        model_type = ModelType.NOISE
    elif prediction == "score":
        model_type = ModelType.SCORE
    else:
        model_type = ModelType.VELOCITY

    if loss_weight == "velocity":
        loss_type = WeightType.VELOCITY
    elif loss_weight == "likelihood":
        loss_type = WeightType.LIKELIHOOD
    else:
        loss_type = WeightType.NONE

    path_choice = {
        "Linear": PathType.LINEAR,
        "GVP": PathType.GVP,
        "VP": PathType.VP,
    }

    path = path_choice[path_type]

    if path in [PathType.VP]:
        train_eps = 1e-5 if train_eps is None else train_eps
        sample_eps = 1e-3 if sample_eps is None else sample_eps
    elif path in [PathType.GVP, PathType.LINEAR] and model_type != ModelType.VELOCITY:
        train_eps = 1e-3 if train_eps is None else train_eps
        sample_eps = 1e-3 if sample_eps is None else sample_eps
    else:  # velocity & [GVP, LINEAR] is stable everywhere
        train_eps = 0
        sample_eps = 0

    # create flow state
    state = Transport(
        model_type=model_type,
        path_type=path,
        loss_type=loss_type,
        train_eps=train_eps,
        sample_eps=sample_eps,
    )

    return state


class Transport:
    """Transport class for handling model transport along different paths with specified loss and model types."""

    def __init__(
        self,
        *,
        model_type: ModelType,
        path_type: PathType,
        loss_type: WeightType,
        train_eps: float,
        sample_eps: float,
    ):
        """Initialize the Transport object with the specified model type, path type, loss type, and epsilon values."""
        path_options = {
            PathType.LINEAR: ICPlan,
            PathType.GVP: GVPCPlan,
            PathType.VP: VPCPlan,
        }

        self.loss_type = loss_type
        self.model_type = model_type
        self.path_sampler = path_options[path_type]()
        self.train_eps = train_eps
        self.sample_eps = sample_eps

    def prior_logp(self, z: th.Tensor) -> th.Tensor:
        """
        Standard multivariate normal prior.

        Assume z is batched
        """
        shape = th.tensor(z.size())
        N = th.prod(shape[1:])

        def _fn(x: th.Tensor) -> th.Tensor:
            return -N / 2.0 * np.log(2 * np.pi) - th.sum(x**2) / 2.0

        return th.vmap(_fn)(z)

    def check_interval(
        self,
        train_eps: float,
        sample_eps: float,
        *,
        diffusion_form: str = "SBDM",
        sde: bool = False,
        reverse: bool = False,
        eval: bool = False,
        last_step_size: float = 0.0,
    ):
        """Check and compute the interval [t0, t1] for the transport process based on the epsilon values, diffusion form, SDE flag, reverse flag, evaluation flag, and last step size."""
        t0 = 0
        t1 = 1
        eps = train_eps if not eval else sample_eps
        if type(self.path_sampler) in [VPCPlan]:
            t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size

        elif (type(self.path_sampler) in [ICPlan, GVPCPlan]) and (
            self.model_type != ModelType.VELOCITY or sde
        ):  # avoid numerical issue by taking a first semi-implicit step
            t0 = (
                eps
                if (diffusion_form == "SBDM" and sde) or self.model_type != ModelType.VELOCITY
                else 0
            )
            t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size

        if reverse:
            t0, t1 = 1 - t0, 1 - t1

        return t0, t1

    def sample(self, x1: th.Tensor):
        """
        Sampling x0 & t based on shape of x1 (if needed).

        Args:
        ----
            x1: data point; [batch, *dim]
        """
        x0 = th.randn_like(x1)
        t0, t1 = self.check_interval(self.train_eps, self.sample_eps)
        t = th.rand((x1.shape[0],)) * (t1 - t0) + t0
        t = t.to(x1)
        return t, x0, x1

    def training_losses(
        self,
        model: Callable[..., Any],
        x1: th.Tensor,
        model_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, th.Tensor]:
        """
        Loss for training the score model.

        Args:
        ----
            model: backbone model; could be score, noise, or velocity; Callable[..., Any]
            x1: datapoint; th.Tensor
            model_kwargs: additional arguments for the model; dict[str, Any] | None
        """
        if model_kwargs is None:
            model_kwargs = {}

        t, x0, x1 = self.sample(x1)
        t, xt, ut = self.path_sampler.plan(t, x0, x1)
        model_output = model(xt, t, **model_kwargs)
        B, *_, C = xt.shape
        assert model_output.size() == (B, *xt.size()[1:-1], C)

        terms: dict[str, th.Tensor] = {}
        terms["pred"] = model_output

        if self.model_type == ModelType.VELOCITY:
            terms["loss"] = mean_flat((model_output - ut) ** 2)
        else:
            _, drift_var = self.path_sampler.compute_drift(xt, t)

            sigma_t, _ = cast(
                "tuple[th.Tensor, th.Tensor]",
                self.path_sampler.compute_sigma_t(expand_t_like_x(t, xt)),
            )
            if self.loss_type in [WeightType.VELOCITY]:
                weight = (drift_var / sigma_t) ** 2
            elif self.loss_type in [WeightType.LIKELIHOOD]:
                weight = drift_var / (sigma_t**2)
            elif self.loss_type in [WeightType.NONE]:
                weight = 1
            else:
                raise NotImplementedError()

            if self.model_type == ModelType.NOISE:
                terms["loss"] = mean_flat(weight * ((model_output - x0) ** 2))
            else:
                terms["loss"] = mean_flat(weight * ((model_output * sigma_t + x0) ** 2))

        return terms

    def get_drift(self) -> Callable[..., Any]:
        """Member function for obtaining the drift of the probability flow ODE."""

        def score_ode(
            x: th.Tensor,
            t: th.Tensor,
            model: Callable[..., Any],
            **model_kwargs: dict[str, Any],
        ) -> th.Tensor:
            drift_mean, drift_var = self.path_sampler.compute_drift(x, t)

            model_output = model(x, t, **model_kwargs)
            return -drift_mean + drift_var * model_output  # by change of variable

        def noise_ode(
            x: th.Tensor,
            t: th.Tensor,
            model: Callable[..., Any],
            **model_kwargs: dict[str, Any],
        ) -> th.Tensor:
            drift_mean, drift_var = self.path_sampler.compute_drift(x, t)
            sigma_t, _ = cast(
                "tuple[th.Tensor, th.Tensor]",
                self.path_sampler.compute_sigma_t(expand_t_like_x(t, x)),
            )
            model_output = model(x, t, **model_kwargs)
            score = model_output / -sigma_t
            return -drift_mean + drift_var * score

        def velocity_ode(
            x: th.Tensor,
            t: th.Tensor,
            model: Callable[..., Any],
            **model_kwargs: dict[str, Any],
        ) -> th.Tensor:
            model_output = model(x, t, **model_kwargs)
            return model_output

        if self.model_type == ModelType.NOISE:
            drift_fn = noise_ode
        elif self.model_type == ModelType.SCORE:
            drift_fn = score_ode
        else:
            drift_fn = velocity_ode

        def body_fn(
            x: th.Tensor,
            t: th.Tensor,
            model: Callable[..., Any],
            **model_kwargs: dict[str, Any],
        ) -> th.Tensor:
            model_output = drift_fn(x, t, model, **model_kwargs)
            assert model_output.shape == x.shape, (
                "Output shape from ODE solver must match input shape"
            )
            return model_output

        return body_fn

    def get_score(
        self,
    ) -> Callable[..., Any]:
        """Member function for obtaining score of x_t = alpha_t * x + sigma_t * eps."""
        if self.model_type == ModelType.NOISE:

            def score_fn(
                x: th.Tensor,
                t: th.Tensor,
                model: Callable[..., Any],
                **kwargs: dict[str, Any],
            ) -> th.Tensor:
                sigma_t, _ = cast(
                    "tuple[th.Tensor, th.Tensor]",
                    self.path_sampler.compute_sigma_t(expand_t_like_x(t, x)),
                )
                return model(x, t, **kwargs) / -sigma_t

        elif self.model_type == ModelType.SCORE:

            def score_fn(
                x: th.Tensor,
                t: th.Tensor,
                model: Callable[..., Any],
                **kwargs: dict[str, Any],
            ) -> th.Tensor:
                return model(x, t, **kwargs)

        elif self.model_type == ModelType.VELOCITY:

            def score_fn(
                x: th.Tensor,
                t: th.Tensor,
                model: Callable[..., Any],
                **kwargs: dict[str, Any],
            ) -> th.Tensor:
                return self.path_sampler.get_score_from_velocity(model(x, t, **kwargs), x, t)

        else:
            raise NotImplementedError()

        return score_fn


class Sampler:
    """Sampler class for the transport model."""

    def __init__(
        self,
        transport: Transport,
    ) -> None:
        """
        Constructor for a general sampler; supporting different sampling methods.

        Args:
        ----
            transport: a transport object specify model prediction & interpolant type
        """
        self.transport = transport
        self.drift = self.transport.get_drift()
        self.score = self.transport.get_score()

    def __get_sde_diffusion_and_drift(
        self,
        *,
        diffusion_form: str = "SBDM",
        diffusion_norm: float = 1.0,
    ):
        def diffusion_fn(x: th.Tensor, t: th.Tensor) -> th.Tensor:
            diffusion = self.transport.path_sampler.compute_diffusion(
                x, t, form=diffusion_form, norm=diffusion_norm
            )

            return diffusion

        def sde_drift(
            x: th.Tensor,
            t: th.Tensor,
            model: Callable[..., Any],
            **kwargs: dict[str, Any],
        ) -> th.Tensor:
            return self.drift(x, t, model, **kwargs) + diffusion_fn(x, t) * self.score(
                x, t, model, **kwargs
            )

        sde_diffusion = diffusion_fn

        return sde_drift, sde_diffusion

    def __get_last_step(
        self,
        sde_drift: Callable[..., th.Tensor],
        *,
        last_step: str | None,
        last_step_size: float,
    ) -> Callable[..., th.Tensor]:
        """Get the last step function of the SDE solver."""
        if last_step is None:

            def last_step_fn(
                x: th.Tensor,
                t: th.Tensor,
                model: Callable[..., Any],
                **model_kwargs: dict[str, Any],
            ) -> th.Tensor:
                return x

        elif last_step == "Mean":

            def last_step_fn(
                x: th.Tensor,
                t: th.Tensor,
                model: Callable[..., Any],
                **model_kwargs: dict[str, Any],
            ) -> th.Tensor:
                return x + sde_drift(x, t, model, **model_kwargs) * last_step_size

        elif last_step == "Tweedie":
            # simple aliasing; the original names were too long
            alpha: Callable[..., Any] = self.transport.path_sampler.compute_alpha_t
            sigma: Callable[..., Any] = self.transport.path_sampler.compute_sigma_t

            def last_step_fn(
                x: th.Tensor,
                t: th.Tensor,
                model: Callable[..., Any],
                **model_kwargs: dict[str, Any],
            ) -> th.Tensor:
                return x / alpha(t)[0][0] + (sigma(t)[0][0] ** 2) / alpha(t)[0][0] * self.score(
                    x, t, model, **model_kwargs
                )

        elif last_step == "Euler":

            def last_step_fn(
                x: th.Tensor,
                t: th.Tensor,
                model: Callable[..., Any],
                **model_kwargs: dict[str, Any],
            ) -> th.Tensor:
                return x + self.drift(x, t, model, **model_kwargs) * last_step_size

        else:
            raise NotImplementedError()

        return last_step_fn

    def sample_sde(
        self,
        *,
        sampling_method: str = "Euler",
        diffusion_form: str = "SBDM",
        diffusion_norm: float = 1.0,
        last_step: str | None = "Mean",
        last_step_size: float = 0.04,
        num_steps: int = 250,
    ):
        """
        Returns a sampling function with given SDE settings.

        Args:
        ----
            sampling_method: type of sampler used in solving the SDE; default to be Euler-Maruyama
            diffusion_form: function form of diffusion coefficient; default to be matching SBDM
            diffusion_norm: function magnitude of diffusion coefficient; default to 1
            last_step: type of the last step; default to identity
            last_step_size: size of the last step; default to match the stride of 250 steps over [0,1]
            num_steps: total integration step of SDE
        """
        if last_step is None:
            last_step_size = 0.0

        sde_drift, sde_diffusion = self.__get_sde_diffusion_and_drift(
            diffusion_form=diffusion_form,
            diffusion_norm=diffusion_norm,
        )

        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            diffusion_form=diffusion_form,
            sde=True,
            eval=True,
            reverse=False,
            last_step_size=last_step_size,
        )

        _sde = sde(
            sde_drift,
            sde_diffusion,
            t0=t0,
            t1=t1,
            num_steps=num_steps,
            sampler_type=sampling_method,
        )

        last_step_fn = self.__get_last_step(
            sde_drift, last_step=last_step, last_step_size=last_step_size
        )

        def _sample(
            init: th.Tensor, model: Callable[..., Any], **model_kwargs: dict[str, Any]
        ) -> list[th.Tensor]:
            xs = _sde.sample(init, model, **model_kwargs)
            ts = th.ones(init.size(0), device=init.device) * t1
            x = last_step_fn(xs[-1], ts, model, **model_kwargs)
            xs.append(x)

            assert len(xs) == num_steps, "Samples does not match the number of steps"

            return xs

        return _sample

    def sample_ode(
        self,
        *,
        sampling_method: str = "dopri5",
        num_steps: int = 50,
        atol: float = 1e-5,  # modified to use default
        rtol: float = 1e-5,
        reverse: bool = False,
    ) -> Callable[..., th.Tensor]:
        """
        Returns a sampling function with given ODE settings.

        Args:
        ----
            sampling_method: type of sampler used in solving the ODE; default to be Dopri5
            num_steps:
                - fixed solver (Euler, Heun): the actual number of integration steps performed
                - adaptive solver (Dopri5): the number of datapoints saved during integration; produced by interpolation
            atol: absolute error tolerance for the solver
            rtol: relative error tolerance for the solver
            reverse: whether solving the ODE in reverse (data to noise); default to False
        """
        if reverse:

            def drift(
                x: th.Tensor,
                t: th.Tensor,
                model: Callable[..., Any],
                **kwargs: dict[str, Any],
            ) -> th.Tensor:
                return self.drift(x, th.ones_like(t) * (1 - t), model, **kwargs)

        else:
            drift = self.drift

        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            sde=False,
            eval=True,
            reverse=reverse,
            last_step_size=0.0,
        )

        _ode = ode(
            drift=drift,
            t0=t0,
            t1=t1,
            sampler_type=sampling_method,
            num_steps=num_steps,
            atol=atol,
            rtol=rtol,
        )

        return _ode.sample

    def sample_ode_likelihood(
        self,
        *,
        sampling_method: str = "dopri5",
        num_steps: int = 50,
        atol: float = 1e-6,
        rtol: float = 1e-3,
    ):
        """
        Returns a sampling function for calculating likelihood with given ODE settings.

        Args:
        ----
            sampling_method: type of sampler used in solving the ODE; default to be Dopri5
            num_steps:
                - fixed solver (Euler, Heun): the actual number of integration steps performed
                - adaptive solver (Dopri5): the number of datapoints saved during integration; produced by interpolation
            atol: absolute error tolerance for the solver
            rtol: relative error tolerance for the solver
        """

        def _likelihood_drift(
            x: th.Tensor,
            t: th.Tensor,
            model: Callable[..., Any],
            **model_kwargs: dict[str, Any],
        ) -> tuple[th.Tensor, th.Tensor]:
            x, _ = x
            eps = th.randint(2, x.size(), dtype=th.float, device=x.device) * 2 - 1
            t = th.ones_like(t) * (1 - t)
            with th.enable_grad():
                x.requires_grad = True
                grad = th.autograd.grad(th.sum(self.drift(x, t, model, **model_kwargs) * eps), x)[0]
                logp_grad = th.sum(grad * eps, dim=tuple(range(1, len(x.size()))))
                drift = self.drift(x, t, model, **model_kwargs)
            return (-drift, logp_grad)

        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            sde=False,
            eval=True,
            reverse=False,
            last_step_size=0.0,
        )

        _ode = ode(
            drift=_likelihood_drift,
            t0=t0,
            t1=t1,
            sampler_type=sampling_method,
            num_steps=num_steps,
            atol=atol,
            rtol=rtol,
        )

        def _sample_fn(
            x: th.Tensor, model: Callable[..., Any], **model_kwargs: dict[str, Any]
        ) -> tuple[th.Tensor, th.Tensor]:
            init_logp = th.zeros(x.size(0)).to(x)
            input = (x, init_logp)
            drift, delta_logp = _ode.sample(input, model, **model_kwargs)
            drift, delta_logp = drift[-1], delta_logp[-1]
            prior_logp = self.transport.prior_logp(drift)
            logp = prior_logp - delta_logp
            return logp, drift

        return _sample_fn
