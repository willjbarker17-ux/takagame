# Football Decision Engine - Technical Brief

## Executive Summary

The Football Decision Engine is a physics-based tactical modeling system for football analysis. It formalizes football positioning and decision-making into computable structures that can evaluate game states, predict outcomes, and support coaching decisions.

**This is a tactical laboratory, not a game engine or entertainment software.**

---

## Core Philosophy

### Starting Assumptions

All players are treated as **physically equal**:
- Same maximum speed (8 m/s)
- Same reaction time (0.25s)
- Same passing/dribbling ability
- No fatigue, psychology, or randomness at the base layer

**The only differentiator is position and space.**

This is a deliberate modeling choice. We understand structure before talent. If the system fails under equal players, it will fail when complexity is added. Talent is treated as a modifier layer, not a foundation.

### Football as Physics

Football is framed as a **dynamic spatial system governed by forces**:

| Aspect | Model |
|--------|-------|
| **Defense** | Attraction-based (defenders pulled toward ball, goal, opponents) |
| **Attack** | Advantage creation through **elimination** |
| **State** | Evaluated via composite scoring function |

---

## Module Architecture

```
src/decision_engine/
├── __init__.py           # Public API exports
├── pitch_geometry.py     # Coordinate system and spatial utilities
├── elimination.py        # Core elimination logic (attacking metric)
├── defense_physics.py    # Attraction-based defensive behavior
├── state_scoring.py      # Game state evaluation system
├── block_models.py       # Defensive block configurations
└── visualizer.py         # Tactical board visualization
```

---

## Core Concepts

### 1. Elimination (The Primary Tactical Currency)

**Definition**: A defender is eliminated if:
1. The ball is past them (positionally, toward the attacking goal)
2. They cannot reach an effective intervention point before the attacker can achieve a more dangerous outcome

**Key Properties**:
- Binary at the moment of evaluation (eliminated or not)
- Based on time-to-intervention, not just distance
- A defender who is goal-side but functionally irrelevant is still eliminated

**Implementation**: `src/decision_engine/elimination.py`

```python
from decision_engine import EliminationCalculator, Player, Position

calculator = EliminationCalculator()
state = calculator.calculate(
    ball_position=Position(20, 5),
    ball_carrier=attacker,
    defenders=[defender1, defender2, defender3],
)

print(f"Eliminated: {state.eliminated_count}/{len(state.defenders)}")
print(f"Ratio: {state.elimination_ratio:.1%}")
```

### 2. Defensive Attraction Physics

Defenders experience multiple attraction forces:

| Force Type | Description | Effect |
|------------|-------------|--------|
| **Ball** | Pull toward ball position | Creates pressing behavior |
| **Goal** | Pull toward own goal | Creates protective depth |
| **Zone** | Pull toward assigned area | Maintains structure |
| **Opponent** | Pull toward dangerous attackers | Creates marking |
| **Teammate** | Repulsion from nearby players | Maintains spacing |
| **Line** | Pull to maintain line height | Creates compactness |

The **equilibrium of these forces** determines optimal positioning.

**Implementation**: `src/decision_engine/defense_physics.py`

```python
from decision_engine import DefensiveForceModel

model = DefensiveForceModel(
    ball_weight=1.0,
    goal_weight=0.6,
    opponent_weight=0.8,
)

forces = model.calculate_forces(
    defender=player,
    ball_position=ball_pos,
    teammates=team,
    opponents=opponents,
)

ideal_position = model.calculate_equilibrium_position(defender, forces)
```

### 3. Game State Scoring

Every game state receives a composite score based on:

| Component | Weight | Description |
|-----------|--------|-------------|
| Elimination | 25% | Number and position of eliminated defenders |
| Proximity | 20% | Distance to goal |
| Angle | 15% | Shooting angle |
| Density | 15% | Space around ball |
| Compactness | 10% | Defensive structure |
| Action | 15% | Available forward options |

**Implementation**: `src/decision_engine/state_scoring.py`

```python
from decision_engine import GameStateEvaluator, GameState

evaluator = GameStateEvaluator()
state = GameState(
    ball_position=Position(25, 0),
    ball_carrier=attacker,
    attackers=[a1, a2, a3],
    defenders=[d1, d2, d3, d4],
)

evaluated = evaluator.evaluate(state)
print(f"Total Score: {evaluated.score.total:.2f}")
print(f"Best Action: {evaluated.available_actions[0].action_type}")
```

### 4. Defensive Block Models

Three primary block configurations:

| Block | Defensive Line | Characteristics |
|-------|---------------|-----------------|
| **Low** | -40m (12.5m from goal) | Compact, protective, minimal pressing |
| **Mid** | -30m (22.5m from goal) | Balanced, controls midfield |
| **High** | -10m (near halfway) | Aggressive, traps opponents |

**Implementation**: `src/decision_engine/block_models.py`

```python
from decision_engine import DefensiveBlock, BlockType

block = DefensiveBlock(BlockType.MID)
positions = block.calculate_positions(
    ball_position=Position(10, 0),
    formation="4-4-2",
)

# Check vulnerability
vuln = block.evaluate_vulnerability(ball_position, defenders)
print(f"Space behind: {vuln['space_behind']:.1%}")
```

---

## Visualization

All outputs render on a tactical board using matplotlib/mplsoccer:

```python
from decision_engine import DecisionEngineVisualizer

viz = DecisionEngineVisualizer()

# Game state view
fig = viz.plot_game_state(state, title="Match Analysis")
viz.save_figure(fig, "output/state.png")

# Elimination analysis
fig = viz.plot_elimination_state(elimination_state, attackers)
viz.save_figure(fig, "output/elimination.png")

# Value heatmap
heatmap = evaluator.generate_value_heatmap(state)
fig = viz.plot_value_heatmap(heatmap)
viz.save_figure(fig, "output/heatmap.png")
```

---

## Data Integration

The engine operates on **coordinate data**, not raw video. It integrates with the existing tracking pipeline:

```
Video → Detection → Tracking → Coordinates → Decision Engine
        (YOLO)     (ByteTrack)  (Homography)     (Analysis)
```

### Input Format

```python
# From tracking output
tracking_frame = {
    "players": [
        {"id": "P1", "x": 25.3, "y": 10.1, "team": "attack"},
        {"id": "P2", "x": -15.0, "y": 5.0, "team": "defense"},
    ],
    "ball": {"x": 22.0, "y": 8.5},
}

# Convert to Decision Engine format
attackers = [
    Player(id=p["id"], position=Position(p["x"], p["y"]))
    for p in tracking_frame["players"] if p["team"] == "attack"
]
```

---

## Use Cases

### 1. Tactical Analysis
- Evaluate positioning quality at any moment
- Identify which defenders are eliminated
- Compare different ball positions

### 2. Opponent Scouting
- Model opponent's defensive block
- Find vulnerabilities in their structure
- Predict where they're weak

### 3. Training Design
- Simulate scenarios to test
- Identify what patterns eliminate defenders
- Design exercises around elimination concepts

### 4. In-Match Analysis
- Real-time state scoring
- Action recommendations
- Block transition timing

---

## Limitations and Assumptions

| Assumption | Implication |
|------------|-------------|
| Equal player speed | No talent-based advantage |
| Instant acceleration | Simplified physics |
| No ball physics | Pass speed is constant |
| Static goalkeeper | Not included in elimination |
| 2D only | No aerial play |

These are **intentional simplifications** for the base model. Layers can be added.

---

## Future Extensions

1. **Player Modifiers**: Add speed/skill coefficients
2. **Probabilistic Outcomes**: Model pass success with uncertainty
3. **Temporal Sequences**: Analyze patterns over time
4. **3D Integration**: Include ball height for aerial play
5. **Machine Learning**: Train from tracking data

---

## File Reference

| File | Purpose | Key Classes |
|------|---------|-------------|
| `pitch_geometry.py` | Spatial math | `Position`, `PitchGeometry` |
| `elimination.py` | Core metric | `EliminationCalculator`, `EliminationState` |
| `defense_physics.py` | Force model | `DefensiveForceModel`, `AttractionForce` |
| `state_scoring.py` | Evaluation | `GameStateEvaluator`, `StateScore` |
| `block_models.py` | Block configs | `DefensiveBlock`, `BlockType` |
| `visualizer.py` | Rendering | `DecisionEngineVisualizer` |

---

## Quick Start

```python
from decision_engine import (
    Position,
    Player,
    GameState,
    GameStateEvaluator,
    DefensiveBlock,
    BlockType,
    DecisionEngineVisualizer,
)

# Create players
attacker = Player(id="A1", position=Position(25, 0))
defenders = [
    Player(id="D1", position=Position(15, -10)),
    Player(id="D2", position=Position(12, 5)),
    Player(id="D3", position=Position(-5, 0)),
]

# Create game state
state = GameState(
    ball_position=Position(25, 0),
    ball_carrier=attacker,
    attackers=[attacker],
    defenders=defenders,
)

# Evaluate
evaluator = GameStateEvaluator()
evaluated = evaluator.evaluate(state)

print(f"Score: {evaluated.score.total:.2f}")
print(f"Eliminated: {evaluated.elimination_state.eliminated_count}")

# Visualize
viz = DecisionEngineVisualizer()
fig = viz.plot_game_state(evaluated, title="Analysis")
fig.savefig("analysis.png")
```

---

## Dependencies

```
numpy
matplotlib
mplsoccer (optional, for enhanced pitch rendering)
```

---

*Last Updated: January 2026*
*Module Location: `src/decision_engine/`*
