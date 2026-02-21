from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class ScoringWeights:
    surprise: float = 0.25
    narrative_arc: float = 0.20
    absurdity: float = 0.30
    volume: float = 0.05
    significance: float = 0.10
    topical: float = 0.10

    @classmethod
    def from_env(cls) -> ScoringWeights:
        return cls(
            surprise=float(os.getenv("WEIGHT_SURPRISE", cls.surprise)),
            narrative_arc=float(os.getenv("WEIGHT_NARRATIVE_ARC", cls.narrative_arc)),
            absurdity=float(os.getenv("WEIGHT_ABSURDITY", cls.absurdity)),
            volume=float(os.getenv("WEIGHT_VOLUME", cls.volume)),
            significance=float(os.getenv("WEIGHT_SIGNIFICANCE", cls.significance)),
            topical=float(os.getenv("WEIGHT_TOPICAL", cls.topical)),
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
