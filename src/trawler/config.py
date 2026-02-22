from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class ScoringWeights:
    surprise: float = 0.12
    narrative_arc: float = 0.08
    volume: float = 0.02
    volume_surprise: float = 0.05
    absurdity: float = 0.10
    significance: float = 0.01
    shareability: float = 0.20
    humor: float = 0.18
    relatability: float = 0.06
    controversy: float = 0.08
    wtf_factor: float = 0.10

    @classmethod
    def from_env(cls) -> ScoringWeights:
        return cls(
            surprise=float(os.getenv("WEIGHT_SURPRISE", cls.surprise)),
            narrative_arc=float(os.getenv("WEIGHT_NARRATIVE_ARC", cls.narrative_arc)),
            volume=float(os.getenv("WEIGHT_VOLUME", cls.volume)),
            volume_surprise=float(os.getenv("WEIGHT_VOLUME_SURPRISE", cls.volume_surprise)),
            absurdity=float(os.getenv("WEIGHT_ABSURDITY", cls.absurdity)),
            significance=float(os.getenv("WEIGHT_SIGNIFICANCE", cls.significance)),
            shareability=float(os.getenv("WEIGHT_SHAREABILITY", cls.shareability)),
            humor=float(os.getenv("WEIGHT_HUMOR", cls.humor)),
            relatability=float(os.getenv("WEIGHT_RELATABILITY", cls.relatability)),
            controversy=float(os.getenv("WEIGHT_CONTROVERSY", cls.controversy)),
            wtf_factor=float(os.getenv("WEIGHT_WTF_FACTOR", cls.wtf_factor)),
        )


@dataclass(frozen=True)
class Config:
    database_url: str = field(
        default_factory=lambda: os.getenv(
            "DATABASE_URL",
            "postgresql://trawler:trawler@localhost:5432/trawler",
        )
    )
    anthropic_api_key: str = field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", "")
    )
    scoring_weights: ScoringWeights = field(
        default_factory=ScoringWeights.from_env
    )


def get_config() -> Config:
    return Config()
