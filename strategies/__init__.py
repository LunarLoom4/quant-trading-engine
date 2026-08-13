from .base import Strategy
from .momentum import MomentumStrategy
from .mean_reversion import MeanReversionStrategy
from .breakout import BreakoutStrategy
from .trend_following import TrendFollowingStrategy

REGISTRY: dict[str, type[Strategy]] = {
    "momentum":        MomentumStrategy,
    "mean_reversion":  MeanReversionStrategy,
    "breakout":        BreakoutStrategy,
    "trend_following": TrendFollowingStrategy,
}


def build_strategy(name: str, config: dict, symbols: list) -> Strategy:
    cls = REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown strategy: '{name}'. Available: {list(REGISTRY)}")
    return cls(config, symbols)


__all__ = [
    "Strategy",
    "MomentumStrategy", "MeanReversionStrategy",
    "BreakoutStrategy", "TrendFollowingStrategy",
    "REGISTRY", "build_strategy",
]
