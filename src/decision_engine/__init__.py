"""
Football Decision Engine

A physics-based tactical modeling system for football analysis.

Core Concepts:
- Elimination: A defender is eliminated when the ball is past them and they
  cannot reach an effective intervention point before the attacker achieves
  a more dangerous outcome.
- Defensive Attraction: Defenders are modeled as being pulled toward ball,
  goal, and key spaces using attraction-based physics.
- State Scoring: Game states are evaluated numerically based on eliminations,
  spatial advantage, and threat proximity.

This is a tactical laboratory, not a game engine.
"""

from .elimination import (
    EliminationCalculator,
    EliminationState,
    DefenderStatus,
)
from .defense_physics import (
    DefensiveForceModel,
    AttractionForce,
    DefensiveShape,
)
from .state_scoring import (
    GameStateEvaluator,
    GameState,
    StateScore,
)
from .block_models import (
    DefensiveBlock,
    BlockType,
    BlockConfiguration,
)
from .pitch_geometry import (
    PitchGeometry,
    PITCH_LENGTH,
    PITCH_WIDTH,
)
from .visualizer import DecisionEngineVisualizer

__all__ = [
    # Elimination
    "EliminationCalculator",
    "EliminationState",
    "DefenderStatus",
    # Defense Physics
    "DefensiveForceModel",
    "AttractionForce",
    "DefensiveShape",
    # State Scoring
    "GameStateEvaluator",
    "GameState",
    "StateScore",
    # Block Models
    "DefensiveBlock",
    "BlockType",
    "BlockConfiguration",
    # Geometry
    "PitchGeometry",
    "PITCH_LENGTH",
    "PITCH_WIDTH",
    # Visualization
    "DecisionEngineVisualizer",
]
