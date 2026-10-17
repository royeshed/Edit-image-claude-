import json
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class StrategyConfig:
    ema_fast: int = 20
    ema_slow: int = 50
    ema_trend: int = 200
    rsi_period: int = 14
    rsi_min: float = 45
    rsi_max: float = 70
    atr_period: int = 14


@dataclass
class RiskConfig:
    risk_per_trade: float = 0.01      # fraction of equity lost if the stop is hit
    atr_stop_mult: float = 2.0        # initial stop distance in ATRs
    reward_risk: float = 2.5          # take-profit distance as a multiple of the stop distance
    trailing_atr_mult: float = 3.0    # trailing stop in ATRs (0 disables)
    max_position_pct: float = 0.95    # never put more than this fraction of equity in one position
    max_daily_loss: float = 0.03      # halt new entries for the day after this drawdown


@dataclass
class CostConfig:
    fee_rate: float = 0.001
    slippage: float = 0.0005


@dataclass
class Config:
    exchange: str = "binance"
    symbol: str = "BTC/USDT"
    timeframe: str = "1h"
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    initial_capital: float = 10_000
    poll_seconds: int = 30
    state_file: str = "state.json"

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        raw = json.loads(Path(path).read_text())
        return cls(
            **{k: v for k, v in raw.items() if k not in ("strategy", "risk", "costs")},
            strategy=StrategyConfig(**raw.get("strategy", {})),
            risk=RiskConfig(**raw.get("risk", {})),
            costs=CostConfig(**raw.get("costs", {})),
        )

    def to_dict(self) -> dict:
        return asdict(self)
