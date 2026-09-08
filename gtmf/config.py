"""Experiment configuration.

Loads a YAML file into an object that fails loudly rather than defaulting. The
failure mode this guards against is specific: a silently-defaulted setting gives
a complete, plausible run of the wrong experiment, and nothing downstream can
tell. So every lookup is explicit, and a key that is missing -- or present but
null where a value is needed -- raises with its own name in the message.
"""

import copy
from datetime import datetime
from pathlib import Path

import yaml

_REQUIRED = object()          # sentinel: distinguishes "no default" from "default None"


class ConfigError(KeyError):
    """A setting is missing, null where it is needed, or inconsistent."""


class Config:
    """A loaded experiment config, addressed by dotted path.

        cfg["gmm.component_sigma"]              -> [0.01, 0.1, 0.3, 0.6]
        cfg.for_sigma("ode.step_size", 0.01)    -> 0.001953125   (by_sigma)
        cfg.for_sigma("ode.step_size", 0.3)     -> 0.015625      (default)
    """

    def __init__(self, data, source=None):
        self.data = copy.deepcopy(data)
        self.source = None if source is None else Path(source)

    # -- loading and saving ------------------------------------------------ #

    @classmethod
    def load(cls, path):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"config not found: {path}")
        with path.open() as fh:
            data = yaml.safe_load(fh)
        if not isinstance(data, dict):
            raise ConfigError(f"{path}: expected a mapping at the top level")
        cfg = cls(data, source=path)
        cfg.validate()
        return cfg

    def dump(self, path):
        """Write the resolved config into a run directory.

        sort_keys=False keeps the authored order, so a diff against the source
        file shows real changes rather than a reshuffle.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as fh:
            yaml.safe_dump(self.data, fh, sort_keys=False, default_flow_style=False)
        return path

    # -- lookup ------------------------------------------------------------- #

    def get(self, dotted, default=_REQUIRED):
        """Fetch by dotted path. Raises unless a default is given explicitly."""
        node = self.data
        for i, part in enumerate(dotted.split(".")):
            if not isinstance(node, dict) or part not in node:
                if default is not _REQUIRED:
                    return default
                where = ".".join(dotted.split(".")[:i + 1])
                raise ConfigError(f"{self._where()}: missing setting '{where}'")
            node = node[part]
        return node

    def __getitem__(self, dotted):
        return self.get(dotted)

    def __contains__(self, dotted):
        """True when the path exists AND holds a value.

        A key present but null counts as absent, which is the useful reading
        here: epsilon_alpha is deliberately null in the production config, and
        "is there something usable?" is what callers actually want to know.
        """
        return self.get(dotted, default=None) is not None

    def for_sigma(self, dotted, sigma):
        """Resolve a {default, by_sigma} mapping for one sigma.

        `by_sigma` wins when it names this sigma, otherwise `default`. A null
        result raises: settings shaped this way (step size, epsilon_alpha) have no
        safe fallback, and epsilon_alpha in particular is left null on purpose so
        a run without a hand-chosen value stops instead of guessing.
        """
        node = self.get(dotted)
        if not isinstance(node, dict):
            # Shorthand: a bare scalar means the same value for every sigma. No
            # ambiguity in that -- one number was written, one number is used.
            node = {"default": node}
        if "default" not in node:
            raise ConfigError(f"{self._where()}: '{dotted}' needs a 'default' key, "
                              f"got keys {sorted(node)}")

        # YAML gives these keys as floats; normalise so 1e-2 and 0.01 match.
        by_sigma = {float(k): v for k, v in (node.get("by_sigma") or {}).items()}
        value = by_sigma.get(float(sigma), node["default"])
        if value is None:
            raise ConfigError(
                f"{self._where()}: '{dotted}' is null for sigma={sigma}. "
                f"Set it under by_sigma, or set a non-null default.")
        return value

    # -- validation ---------------------------------------------------------- #

    def validate(self):
        """Check what can be checked without running anything."""
        for key in ("experiment", "gmm.component_sigma", "time.num_points",
                    "time.t_min", "time.t_max", "monte_carlo.num_query_states",
                    "monte_carlo.num_probes", "monte_carlo.scheme",
                    "ode.integrator", "ode.step_size", "output.output_dir"):
            self.get(key)

        sigmas = self.get("gmm.component_sigma")
        if not isinstance(sigmas, list) or not sigmas:
            raise ConfigError(f"{self._where()}: 'gmm.component_sigma' must be a "
                              f"non-empty list, got {sigmas!r}")
        if any(s <= 0 for s in sigmas):
            raise ConfigError(f"{self._where()}: sigmas must be positive, got {sigmas}")

        # A by_sigma key that matches no swept sigma is silently ignored otherwise,
        # and the sigma it was meant for quietly runs on the default -- a plausible
        # run at the wrong step size. Typos here must not be survivable.
        swept = {float(s) for s in sigmas}
        for dotted in ("ode.step_size", "monte_carlo.epsilon_alpha"):
            node = self.get(dotted, default=None)
            if isinstance(node, dict):
                for key in (node.get("by_sigma") or {}):
                    if float(key) not in swept:
                        raise ConfigError(
                            f"{self._where()}: '{dotted}.by_sigma' names sigma={key}, "
                            f"which is not in gmm.component_sigma {sorted(swept)}")

        if self.get("time.t_min") >= self.get("time.t_max"):
            raise ConfigError(f"{self._where()}: time.t_min must be below t_max")
        if self.get("time.num_points") < 2:
            raise ConfigError(f"{self._where()}: time.num_points must be at least 2")
        for key in ("monte_carlo.num_query_states", "monte_carlo.num_probes"):
            if self.get(key) < 1:
                raise ConfigError(f"{self._where()}: '{key}' must be at least 1")
        if self.get("monte_carlo.scheme") not in ("one_sided", "central"):
            raise ConfigError(f"{self._where()}: monte_carlo.scheme must be "
                              f"'one_sided' or 'central'")
        if self.get("ode.integrator") not in ("rk4", "euler"):
            raise ConfigError(f"{self._where()}: ode.integrator must be 'rk4' or 'euler'")
        # Optional with a default, unlike the keys above, so it is checked only
        # for a bad VALUE -- but checked here rather than at first use, so a typo
        # stops the run before 1.3 GB of centres is read.
        if self.get("monte_carlo.rho_source", default="exact") not in ("exact",
                                                                      "empirical"):
            raise ConfigError(f"{self._where()}: monte_carlo.rho_source must be "
                              f"'exact' or 'empirical'")
        if self.get("data.source", default="latents") not in ("latents", "synthetic"):
            raise ConfigError(f"{self._where()}: data.source must be 'latents' or "
                              f"'synthetic'")
        return self

    # -- run layout ----------------------------------------------------------- #

    def run_dir(self, root=None, stamp=None):
        """results/<experiment>_<timestamp>/ -- one invocation, one directory.

        The resolved config and the log belong to the run; each swept sigma gets a
        subdirectory under it (see `sigma_dir`).
        """
        root = Path(self.get("output.output_dir") if root is None else root)
        stamp = stamp or datetime.now().strftime("%Y%m%d-%H%M%S")
        return root / f"{self.get('experiment')}_{stamp}"

    @staticmethod
    def sigma_dir(run_dir, sigma):
        """One subdirectory per swept sigma, named so they sort in order."""
        return Path(run_dir) / f"sigma_{float(sigma):g}"

    def _where(self):
        return str(self.source) if self.source else "config"

    def __repr__(self):
        return (f"Config({self.get('experiment', default='?')!r}, "
                f"sigmas={self.get('gmm.component_sigma', default='?')}, "
                f"source={self._where()})")
