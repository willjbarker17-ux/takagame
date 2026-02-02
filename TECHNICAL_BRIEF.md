# Marshall Soccer AI Platform — Technical Brief

This document provides technical details on how both parts of the system work under the hood.

---

# Part 1: Marshall Intelligence Hub

## Overview

The Hub is a **Retrieval-Augmented Generation (RAG)** system that connects Marshall's scattered data into one queryable knowledge base.

```
Data Sources → Ingestion → Vector Database → Query Interface → LLM → Response
```

## How RAG Works

Traditional AI chatbots generate responses from their training data alone. RAG is different:

1. **User asks a question** ("What did we say about Kentucky's press last time we played them?")
2. **System searches** the vector database for relevant content (meeting transcripts, scouting docs, Wyscout notes)
3. **Retrieves** the most relevant chunks of text
4. **Sends to LLM** with the question + retrieved context
5. **LLM generates** a response grounded in Marshall's actual data

**Why this matters:** The AI doesn't hallucinate or give generic answers. It answers from our specific documents, meetings, and data.

## Technical Components

### 1. Data Ingestion Pipeline

Each data source requires a specific ingestion approach:

| Source | Format | Ingestion Method |
|--------|--------|------------------|
| Wyscout | API/CSV exports | Parse match events, player stats, team reports into structured records |
| GPS | CSV exports | Parse 240+ metrics per session, normalize timestamps, link to matches |
| Game Plans | PDF/Canva | Extract text via OCR if needed, parse into sections |
| Meeting Recordings | Audio/Video | Transcribe via Whisper, segment by speaker/topic |
| Recruitment | Spreadsheets/Notes | Parse evaluations, link to player profiles |
| Medical/Schedule | Various | Structured parsing based on format |

**Processing steps:**
```
Raw Data → Parse/Extract → Clean/Normalize → Chunk → Embed → Store
```

### 2. Text Chunking

Documents are split into chunks (typically 500-1000 tokens) with overlap to preserve context. Each chunk becomes a searchable unit.

**Chunking strategies:**
- **Meeting transcripts:** Split by speaker turn or topic shift
- **Scouting reports:** Split by section (formation, tendencies, set pieces)
- **Match data:** Group by match, then by phase of play

### 3. Vector Embeddings

Each chunk is converted to a **vector embedding** — a numerical representation that captures semantic meaning.

```python
# Example using OpenAI embeddings
from openai import OpenAI
client = OpenAI()

response = client.embeddings.create(
    input="Kentucky uses a 4-1-4-1 high press with the striker triggering",
    model="text-embedding-3-small"
)
vector = response.data[0].embedding  # 1536-dimensional vector
```

**Why vectors:** Similar content has similar vectors. "Kentucky's high press" and "their pressing triggers" will be close in vector space, even if the exact words differ.

### 4. Vector Database

Embeddings are stored in a vector database optimized for similarity search.

**Options:**
- **Chroma** — Open source, easy to set up, good for prototyping
- **Pinecone** — Managed service, scales well, production-ready
- **Weaviate** — Open source with hybrid search capabilities

**Storage structure:**
```
{
    "id": "meeting_2024_03_15_chunk_23",
    "vector": [0.023, -0.156, 0.089, ...],  # 1536 dimensions
    "metadata": {
        "source": "staff_meeting",
        "date": "2024-03-15",
        "speakers": ["Coach A", "Coach B"],
        "topics": ["Kentucky", "high press", "set pieces"]
    },
    "text": "Looking at Kentucky's press, they trigger with the striker..."
}
```

### 5. Query Processing

When a user asks a question:

```python
# 1. Embed the query
query_vector = embed("What did we discuss about Kentucky's press?")

# 2. Search vector database for similar chunks
results = vector_db.query(
    vector=query_vector,
    top_k=10,
    filter={"source": {"$in": ["staff_meeting", "scouting_report"]}}
)

# 3. Retrieve text from top matches
context_chunks = [r.text for r in results]

# 4. Build prompt with retrieved context
prompt = f"""
Based on the following Marshall staff discussions and documents:

{context_chunks}

Answer this question: What did we discuss about Kentucky's press?
"""

# 5. Generate response via LLM
response = llm.generate(prompt)
```

### 6. LLM Integration

The system uses Claude API (or similar) for generation:

```python
import anthropic

client = anthropic.Anthropic()

message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    system="You are Marshall Soccer's internal knowledge assistant. Answer based only on the provided context. If the context doesn't contain relevant information, say so.",
    messages=[
        {"role": "user", "content": prompt_with_context}
    ]
)
```

**Key settings:**
- **System prompt:** Instructs the AI to answer only from provided context
- **Temperature:** Low (0.1-0.3) for factual accuracy
- **Context window:** 100K+ tokens allows including many relevant chunks

## Cross-Source Pattern Recognition

The real power is linking data across sources that don't naturally connect.

**Example query:** "How did our players perform physically against pressing teams this season?"

**What happens:**
1. System identifies "pressing teams" from Wyscout/scouting data
2. Pulls GPS load data from matches against those opponents
3. Retrieves any meeting discussions about physical preparation
4. Synthesizes across all three sources

**This requires:**
- Consistent entity linking (match IDs, player IDs, opponent names)
- Metadata tagging during ingestion
- Query understanding that spans domains

## Meeting Capture Setup

Meetings are the highest-signal data source but require capture infrastructure.

**Recording options:**
1. **Screen recording** (OBS/Loom) during Zoom/film sessions
2. **HDMI capture dongle** for in-room presentations
3. **Audio recorder** for tactical discussions

**Transcription pipeline:**
```
Audio/Video → Whisper (local or API) → Raw transcript → Speaker diarization → Cleaned transcript → Chunking → Embedding
```

**Whisper integration:**
```python
import whisper

model = whisper.load_model("large-v2")
result = model.transcribe("meeting_2024_03_15.mp4")
transcript = result["text"]
segments = result["segments"]  # Timestamped segments
```

**Speaker identification:**
- Manual tagging during review
- Or voice fingerprinting for automatic speaker labels

## Learning Over Time

The Hub improves as it accumulates more Marshall-specific data:

1. **More context:** Each meeting, each match adds to the knowledge base
2. **Pattern emergence:** Trends only visible across months/years of data
3. **Feedback loop:** Staff corrections improve retrieval quality
4. **Custom terminology:** System learns Marshall's vocabulary and concepts

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     DATA SOURCES                                 │
├──────────┬──────────┬──────────┬──────────┬──────────┬─────────┤
│ Wyscout  │   GPS    │  Game    │ Meeting  │ Recruit  │  Other  │
│  API     │ Exports  │  Plans   │ Records  │  Notes   │         │
└────┬─────┴────┬─────┴────┬─────┴────┬─────┴────┬─────┴────┬────┘
     │          │          │          │          │          │
     ▼          ▼          ▼          ▼          ▼          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    INGESTION PIPELINE                            │
│  Parse → Clean → Normalize → Chunk → Embed → Store              │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    VECTOR DATABASE                               │
│  (Chroma / Pinecone / Weaviate)                                 │
│  - Embeddings for semantic search                               │
│  - Metadata for filtering                                       │
│  - Full text for retrieval                                      │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    QUERY INTERFACE                               │
│  User Question → Embed → Search → Retrieve → Build Prompt       │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LLM (Claude API)                              │
│  Context + Question → Generate Response → Return to User        │
└─────────────────────────────────────────────────────────────────┘
```

## Tech Stack Summary

| Component | Technology | Purpose |
|-----------|------------|---------|
| Embeddings | OpenAI text-embedding-3-small | Convert text to vectors |
| Vector DB | Chroma (dev) / Pinecone (prod) | Store and search embeddings |
| LLM | Claude API | Generate responses |
| Transcription | Whisper large-v2 | Convert audio to text |
| Backend | Python + FastAPI | API layer |
| Frontend | Simple web interface | Staff interaction |

---

# Part 2: Tracking + Decision Engine

## Overview

Part 2 has two layers:
1. **Tracking Layer:** Extract player/ball coordinates from video
2. **Decision Engine Layer:** Convert coordinates into tactical measurements

```
Video → Detection → Tracking → Calibration → Coordinates → Decision Engine → Insights
```

---

## Layer 1: Tracking System

### Current Codebase: 36,000+ Lines

```
src/
├── detection/           # Player and ball detection
│   ├── player_detector.py    # YOLOv8-based detection
│   ├── ball_detector.py      # Specialized ball detection
│   ├── detr_detector.py      # Transformer-based detection (advanced)
│   ├── team_classifier.py    # Jersey color classification
│   └── hybrid_detector.py    # Multi-method combination
│
├── tracking/            # Identity maintenance across frames
│   └── tracker.py            # ByteTrack integration
│
├── homography/          # Pixel-to-pitch coordinate conversion
│   ├── calibration.py        # Manual calibration tools
│   ├── auto_calibration.py   # Automatic keypoint-based calibration
│   ├── field_model.py        # FIFA pitch geometry (57+ keypoints)
│   ├── keypoint_detector.py  # Deep learning keypoint detection
│   └── rotation_handler.py   # Camera movement handling
│
└── training/            # Model training infrastructure
    ├── configs/              # 6 YAML configuration files
    ├── datasets/             # SoccerNet, SkillCorner, Synthetic loaders
    ├── utils/                # Losses, metrics, callbacks
    └── train_*.py            # Training scripts for each model
```

### Detection: How We Find Players

**Primary method: YOLOv8**

```python
from ultralytics import YOLO

class PlayerDetector:
    def __init__(self, model_path="yolov8x.pt", confidence=0.3):
        self.model = YOLO(model_path)
        self.confidence = confidence

    def detect(self, frame):
        results = self.model(frame, conf=self.confidence)
        detections = []
        for box in results[0].boxes:
            if box.cls == 0:  # Person class
                detections.append(Detection(
                    bbox=box.xyxy[0].tolist(),
                    confidence=box.conf.item(),
                    class_id=0
                ))
        return detections
```

**Current state:**
- ✅ Can detect players in individual frames
- ✅ Works well with clear, unoccluded players
- ⚠️ Struggles with crowded areas, overlapping players
- ⚠️ Ball detection unreliable in traffic

**Advanced method: DETR (Detection Transformer)**

Architecture defined but not yet trained. Uses attention mechanisms to handle crowded scenes better than YOLO.

### Tracking: How We Maintain Identity

**ByteTrack integration:**

```python
import supervision as sv

class PlayerTracker:
    def __init__(self, track_buffer=30, match_thresh=0.8):
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.5,
            lost_track_buffer=track_buffer,
            minimum_matching_threshold=match_thresh,
            frame_rate=25
        )
        self.tracks = {}

    def update(self, detections):
        # Update tracker with new detections
        tracked = self.tracker.update_with_detections(detections)

        # Maintain track history
        for track in tracked:
            if track.track_id not in self.tracks:
                self.tracks[track.track_id] = Track(track.track_id)
            self.tracks[track.track_id].update(track.bbox)

        return tracked
```

**Current state:**
- ✅ Basic tracking works on stable footage
- ⚠️ Loses identity through occlusions
- ⚠️ Camera cuts require re-identification
- ❌ Not reliable across full 90 minutes yet

### Calibration: Converting Pixels to Pitch Coordinates

This is the **main technical challenge** with Wyscout footage.

**The problem:**
- Wyscout uses wide-angle cameras
- Camera angle shifts during play
- Need to know where each pixel corresponds to on the pitch

**Manual calibration (working):**

```python
class HomographyEstimator:
    def __init__(self):
        self.keypoints_pixel = []   # Clicked points in image
        self.keypoints_world = []   # Corresponding pitch coordinates

    def add_correspondence(self, pixel_xy, world_xy):
        self.keypoints_pixel.append(pixel_xy)
        self.keypoints_world.append(world_xy)

    def compute_homography(self):
        # Requires minimum 4 point pairs
        if len(self.keypoints_pixel) < 4:
            raise ValueError("Need at least 4 keypoints")

        # Compute 3x3 transformation matrix
        H, _ = cv2.findHomography(
            np.array(self.keypoints_pixel),
            np.array(self.keypoints_world),
            cv2.RANSAC
        )
        return H

    def transform_point(self, pixel_xy, H):
        # Convert pixel to pitch coordinates
        px = np.array([pixel_xy[0], pixel_xy[1], 1])
        world = H @ px
        return world[:2] / world[2]  # Normalize
```

**Automatic calibration (in development):**

```python
class AutoCalibrator:
    def __init__(self, keypoint_model, field_model):
        self.keypoint_detector = keypoint_model  # Neural network
        self.field_model = field_model  # FIFA pitch geometry

    def calibrate(self, frame):
        # 1. Detect pitch keypoints using neural network
        detected_keypoints = self.keypoint_detector.detect(frame)

        # 2. Match to known world coordinates
        correspondences = self.match_to_field_model(detected_keypoints)

        # 3. RANSAC-based homography estimation
        H, inliers = cv2.findHomography(
            correspondences.pixels,
            correspondences.world,
            cv2.RANSAC,
            ransacReprojThreshold=5.0
        )

        # 4. Quality assessment
        quality = self.assess_quality(H, correspondences, inliers)

        return AutoCalibrationResult(H, quality)
```

**Current state:**
- ✅ Manual calibration works with static camera angle
- ⚠️ Automatic calibration code exists but needs training data
- ❌ Handling angle shifts during play is the main challenge

**Field model (57+ keypoints defined):**

```python
class FootballPitchModel:
    # FIFA standard: 105m x 68m
    LENGTH = 105.0
    WIDTH = 68.0

    KEYPOINTS = {
        # Corners
        "corner_top_left": (-52.5, 34.0),
        "corner_top_right": (52.5, 34.0),
        "corner_bottom_left": (-52.5, -34.0),
        "corner_bottom_right": (52.5, -34.0),

        # Halfway line
        "halfway_top": (0, 34.0),
        "halfway_bottom": (0, -34.0),
        "center_spot": (0, 0),

        # Penalty boxes (16.5m from goal line, 40.3m wide)
        "penalty_box_left_top": (-52.5 + 16.5, 20.15),
        # ... 57+ total keypoints
    }
```

### Training Infrastructure

**6 models with full training setup:**

| Model | Purpose | Status |
|-------|---------|--------|
| Homography | Keypoint detection for calibration | Training script ready |
| Baller2Vec | Trajectory embeddings | Training script ready |
| Ball3D | 3D ball tracking with physics | Stub |
| Re-ID | Player re-identification | Stub |
| DETR | Transformer detection | Stub |
| GNN | Tactical graph networks | Stub |

**Dataset loaders implemented:**
- **SoccerNet:** 29 pitch keypoints, player trajectories, bounding boxes
- **SkillCorner:** Professional tracking data (10Hz)
- **Synthetic:** Physics-based ball trajectories, player movements

**Training utilities:**
- 9 loss functions (keypoint, reprojection, trajectory, contrastive, Hungarian matching, physics constraints)
- 7 metric calculators (reprojection error, trajectory ADE/FDE, detection mAP)
- 7 callbacks (checkpointing, early stopping, LR scheduling, W&B logging)

---

## Layer 2: Decision Engine

### Module Architecture

```
src/decision_engine/
├── pitch_geometry.py     # Coordinate system (FIFA 105m x 68m)
├── elimination.py        # Core attacking metric
├── defense_physics.py    # Force-based positioning model
├── state_scoring.py      # Composite game state evaluation
├── block_models.py       # Defensive block configurations
└── visualizer.py         # Tactical board rendering
```

### Module 1: Pitch Geometry

Defines the coordinate system and spatial utilities.

```python
# Origin at center, +x toward attacking goal, +y toward top touchline
HALF_LENGTH = 52.5  # meters
HALF_WIDTH = 34.0   # meters

class Position:
    x: float  # -52.5 to +52.5
    y: float  # -34.0 to +34.0

    def distance_to(self, other: Position) -> float:
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

    def angle_to(self, other: Position) -> float:
        return math.atan2(other.y - self.y, other.x - self.x)

class PitchGeometry:
    def is_in_penalty_area(self, pos: Position, attacking: bool) -> bool:
        ...

    def distance_to_goal(self, pos: Position) -> float:
        ...

    def shooting_angle(self, pos: Position) -> float:
        # Angle subtended by goal posts from position
        ...
```

### Module 2: Elimination Calculator

**The core tactical metric.**

A defender is **eliminated** when they cannot affect the play in time, regardless of their nominal position.

```python
class Player:
    id: str
    position: Position
    velocity: Velocity
    max_speed: float = 8.0      # m/s (elite sprint)
    reaction_time: float = 0.25  # seconds

    def time_to_position(self, target: Position) -> float:
        distance = self.position.distance_to(target)
        return self.reaction_time + distance / self.max_speed

class EliminationCalculator:
    def calculate(self, ball_position, ball_carrier, defenders) -> EliminationState:
        results = []
        for defender in defenders:
            # 1. Is defender goal-side of ball?
            is_goal_side = self._is_goal_side(defender, ball_position)

            # 2. Find optimal intervention point
            intervention_point = self._find_intervention_point(
                defender.position, ball_position, goal_position
            )

            # 3. Time for defender to reach intervention
            defender_time = defender.time_to_position(intervention_point)

            # 4. Time for attacker to progress past intervention
            attacker_time = self._calculate_danger_time(
                ball_position, ball_carrier, intervention_point
            )

            # 5. Determine status
            if not is_goal_side or attacker_time < defender_time:
                status = DefenderStatus.ELIMINATED
            else:
                status = DefenderStatus.ACTIVE

            results.append(EliminationResult(defender, status, ...))

        return EliminationState(ball_position, ball_carrier, results)
```

**Output:**
```python
state.eliminated_count      # 3
state.active_count          # 4
state.elimination_ratio     # 0.43
state.get_eliminated()      # [Player, Player, Player]
```

### Module 3: Defensive Force Model

Models defensive positioning as equilibrium of attraction forces.

```python
class DefensiveForceModel:
    def __init__(
        self,
        ball_weight=1.0,        # Pressing intensity
        goal_weight=0.6,        # Protective depth
        zone_weight=0.4,        # Structural discipline
        opponent_weight=0.8,    # Marking tightness
        teammate_repulsion=0.3, # Spacing
        line_weight=0.5,        # Compactness
    ):
        self.weights = {...}

    def calculate_forces(self, defender, ball_pos, teammates, opponents):
        forces = []

        # Ball attraction (pressing)
        ball_force = self._ball_attraction(defender, ball_pos)
        forces.append(ball_force * self.ball_weight)

        # Goal attraction (protection)
        goal_force = self._goal_attraction(defender)
        forces.append(goal_force * self.goal_weight)

        # Opponent attraction (marking)
        for opp in opponents:
            opp_force = self._opponent_attraction(defender, opp)
            forces.append(opp_force * self.opponent_weight)

        # Teammate repulsion (spacing)
        for tm in teammates:
            tm_force = self._teammate_repulsion(defender, tm)
            forces.append(tm_force * self.teammate_repulsion)

        return forces

    def calculate_equilibrium_position(self, defender, forces):
        # Sum all force vectors to find ideal position
        net_force = sum(forces)
        return defender.position + net_force
```

**Key insight: Weights are tunable.**

```python
# Aggressive pressing team
pressing_model = DefensiveForceModel(ball_weight=1.5, goal_weight=0.3)

# Deep-sitting team
defensive_model = DefensiveForceModel(ball_weight=0.5, goal_weight=1.0)

# Tune to match opponent's actual behavior
opponent_model = DefensiveForceModel(
    ball_weight=0.8,   # Calibrated from
    goal_weight=0.7,   # watching their
    opponent_weight=1.2 # film
)
```

### Module 4: Game State Evaluator

Produces a composite score for any game moment.

```python
class GameStateEvaluator:
    def evaluate(self, state: GameState) -> EvaluatedState:
        # Calculate component scores (each 0-1)
        elimination_score = self._score_elimination(state)  # 25%
        proximity_score = self._score_proximity(state)      # 20%
        angle_score = self._score_angle(state)              # 15%
        density_score = self._score_density(state)          # 15%
        compactness_score = self._score_compactness(state)  # 10%
        action_score = self._score_actions(state)           # 15%

        total = (
            0.25 * elimination_score +
            0.20 * proximity_score +
            0.15 * angle_score +
            0.15 * density_score +
            0.10 * compactness_score +
            0.15 * action_score
        )

        return EvaluatedState(state, StateScore(
            elimination_score, proximity_score, angle_score,
            density_score, compactness_score, action_score,
            total
        ))
```

**Use case: Comparing decisions**

```python
# Actual play: pass to winger
actual_state = evaluator.evaluate(state_after_winger_pass)
print(f"Winger pass: {actual_state.score.total:.3f}")  # 0.42

# Alternative: switch to weak side
simulated_state = evaluator.evaluate(state_after_switch)
print(f"Switch play: {simulated_state.score.total:.3f}")  # 0.58

# The switch was +0.16 better
```

### Module 5: Defensive Block Models

Defines standard defensive configurations.

```python
class BlockType(Enum):
    LOW_BLOCK = "low"          # ~12.5m from goal
    MID_BLOCK = "mid"          # ~22.5m from goal
    HIGH_BLOCK = "high"        # Near halfway line
    ULTRA_LOW = "ultra_low"    # Park the bus
    ULTRA_HIGH = "ultra_high"  # Full pitch press

class DefensiveBlock:
    def __init__(self, block_type: BlockType):
        self.config = BLOCK_CONFIGS[block_type]

    def calculate_positions(self, ball_position, formation="4-4-2"):
        # Return ideal positions for each player in formation
        positions = {}

        # Defensive line height based on block type
        defensive_line_y = self.config.defensive_line_height

        # Adjust based on ball position
        if ball_position.x > 0:  # Ball in our half
            defensive_line_y -= self.config.ball_side_shift

        # Calculate positions for 4 defenders, 4 midfielders, 2 forwards
        ...

        return positions

    def evaluate_vulnerability(self, ball_pos, actual_positions):
        return {
            "space_behind": self._calculate_space_behind(),
            "gaps_between_lines": self._calculate_line_gaps(),
            "exposed_flanks": self._check_flank_exposure(),
        }
```

### Module 6: Visualizer

Renders analysis on tactical boards.

```python
class DecisionEngineVisualizer:
    def plot_game_state(self, evaluated_state, title=""):
        fig, ax = plt.subplots(figsize=(12, 8))

        # Draw pitch
        self._draw_pitch(ax)

        # Draw players (color by elimination status)
        for result in evaluated_state.elimination_state.defenders:
            color = 'red' if result.is_eliminated else 'blue'
            self._draw_player(ax, result.defender, color)

        # Draw ball
        self._draw_ball(ax, evaluated_state.ball_position)

        # Add score overlay
        self._add_score_overlay(ax, evaluated_state.score)

        return fig

    def plot_value_heatmap(self, state, resolution=50):
        # Generate heatmap of state scores across pitch
        ...
```

---

## End-to-End Pipeline

### From Video to Insights

```
┌─────────────────────────────────────────────────────────────────┐
│                        VIDEO INPUT                               │
│                    (Wyscout match footage)                       │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                        DETECTION                                 │
│  YOLOv8 detects players + ball in each frame                    │
│  Output: List of bounding boxes with confidence                 │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                       CALIBRATION                                │
│  Homography maps pixel coordinates to pitch coordinates         │
│  Challenge: Handle camera angle changes in Wyscout footage      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                        TRACKING                                  │
│  ByteTrack maintains consistent player IDs across frames        │
│  Challenge: Occlusions, camera cuts, crowded areas              │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  WORLD COORDINATES                               │
│  22 players + ball with (x, y) positions on pitch               │
│  Validation: Compare against GPS ground truth                   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   DECISION ENGINE                                │
│                                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ Elimination │  │   Force     │  │   State     │             │
│  │ Calculator  │  │   Model     │  │  Evaluator  │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
│                                                                  │
│  Output: Elimination counts, structure metrics, state scores    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                       INSIGHTS                                   │
│  - Which defenders were eliminated on each play                 │
│  - Where structure broke down                                   │
│  - xG value of alternative decisions                            │
│  - Trends across match/season                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Current State Summary

### What's Working ✅

| Component | Status | Notes |
|-----------|--------|-------|
| Decision Engine (all 6 modules) | ✅ Complete | Fully implemented, tested |
| Player detection (YOLOv8) | ✅ Working | Good on clear frames |
| Manual calibration | ✅ Working | Requires human input |
| ByteTrack integration | ✅ Working | Basic tracking functional |
| Training infrastructure | ✅ Ready | 6 configs, 4 datasets, full utils |

### What Needs Work ⚠️

| Component | Status | Challenge |
|-----------|--------|-----------|
| Automatic calibration | ⚠️ Code exists | Needs training data for keypoint detection |
| Full-field tracking | ⚠️ Partial | Loses identity through occlusions |
| Ball tracking | ⚠️ Partial | Unreliable in traffic |
| Camera angle handling | ⚠️ Main challenge | Wyscout footage shifts during play |
| GPS validation | ⚠️ Not started | Need to compare against ground truth |

### What's Not Started ❌

| Component | Status | Dependency |
|-----------|--------|------------|
| DETR training | ❌ Stub | Needs SoccerNet data |
| Re-ID training | ❌ Stub | Needs tracking data |
| Ball3D training | ❌ Stub | Needs ball annotations |
| GNN training | ❌ Stub | Needs formation labels |

---

## Development Sequence

### Phase 1: Calibration (Month 1)
- Train keypoint detector on SoccerNet data
- Achieve consistent automatic calibration
- Handle camera angle changes

### Phase 2: Tracking (Month 2)
- Improve ByteTrack parameters
- Add re-identification for occlusion recovery
- Validate against GPS data

### Phase 3: Validation (Month 3)
- Run full matches through pipeline
- Compare coordinates against GPS
- Establish accuracy baselines

### Phase 4: Decision Engine Integration (Months 4-5)
- Connect validated tracking to engine
- Calibrate force weights with coaching staff
- Confirm face validity (engine flags what coaches see)

---

## Dependencies

```
# Detection
ultralytics          # YOLOv8
torch                # PyTorch
torchvision          # Image processing

# Tracking
supervision          # ByteTrack wrapper

# Homography
opencv-python        # Calibration, transforms
numpy                # Linear algebra

# Decision Engine
numpy                # Calculations
matplotlib           # Visualization
mplsoccer           # Pitch rendering

# Training
wandb                # Experiment tracking
albumentations       # Augmentation

# Hub (Part 1)
openai               # Embeddings
anthropic            # Claude API
chromadb             # Vector database
whisper              # Transcription
```

---

*This document accompanies MARSHALL_SOCCER_AI_PITCH.md*
*Last updated: February 2026*
