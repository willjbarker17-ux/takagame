# Decision Engine Development Plan

## The Goal

Get players into higher xG shooting zones by identifying the best available option at any moment.

The engine doesn't replace coaching judgment — it provides objective measurement. Coaches see patterns humans miss. The engine sees every frame and quantifies every option.

---

## What We Have

**Five working modules:**

| Module | What It Does |
|--------|--------------|
| Elimination Calculator | Determines which defenders are functionally out of play based on time-to-position |
| Defensive Force Model | Physics-based model of where defenders should be, identifies structural gaps |
| Game State Evaluator | Scores any position 0.0-1.0 across six dimensions |
| Block Models | Templates for defensive heights (low block, mid block, high press) |
| Visualizer | Interactive tactical board for manual analysis |

**The math works.** Each module runs independently, produces deterministic outputs, and can be tuned.

**What's missing:** Real game data flowing through the system, validation against coaching judgment, and learning from outcomes.

---

## The Problem

The engine uses physics to evaluate options:
- Time for attacker to reach a position
- Time for defender to intercept
- Resulting xG if the ball gets there

Physics tells us what's *possible*. It doesn't tell us what *works*.

A through ball might be physically optimal but consistently fails because:
- The receiver doesn't expect it
- The passer lacks the technique
- The timing is off by 0.2 seconds

**The gap between physics and reality is where learning happens.**

---

## Development Approach

### Phase 1: Connect the Pipe

**Goal:** Tracking data flows into the decision engine, produces per-frame analysis.

**Work:**
1. Modify `src/main.py` frame loop to extract player coordinates from tracker
2. Transform pixel coordinates to pitch coordinates via homography
3. Create `Player` objects with position, velocity, team
4. Feed into `GameStateEvaluator` for each frame
5. Store results (position score, elimination state, available actions)

**Output:** A match produces a JSON file with frame-by-frame tactical analysis.

**Validation:** Run on 2-3 matches. Confirm coordinates look correct. Spot-check a few moments manually.

**Blockers:** Tracking accuracy. If player positions are wrong, analysis is meaningless. This phase depends on tracking layer being reliable.

---

### Phase 2: Face Validity

**Goal:** Coaches agree the engine identifies meaningful moments.

**Work:**
1. Process 3 full matches through the pipe
2. Engine flags "key moments" — high position score changes, elimination breakthroughs, defensive breakdowns
3. Present flagged moments to coaching staff via video clips
4. Document: "Engine says this was significant. Do you agree?"

**Output:** Qualitative validation. Coaches either trust the output or don't.

**Adjustment:** If engine misses moments coaches care about, adjust evaluation weights. If engine flags noise, tighten thresholds.

**Success criteria:** 80%+ agreement on flagged moments being tactically significant.

---

### Phase 3: Game Model Integration

**Goal:** Engine speaks Marshall's language.

**Work:**
1. Define moment types from game model:
   - BGZ buildup (building from back)
   - High Loss counter-press
   - Exploit Space AROUND
   - +1 overloads
   - Transition moments
2. Create labeling schema — each moment gets a type
3. Label 10+ matches with moment types
4. Engine learns: "When we're in BGZ buildup, these options tend to work"

**Output:** Engine can identify "This is a counter-press situation" and evaluate options through that lens.

**Why this matters:** A pass that's optimal in BGZ buildup might be wrong in transition. Context changes evaluation.

---

### Phase 4: Outcome Learning

**Goal:** Engine learns from what actually happens.

**Work:**
1. Connect decisions to outcomes:
   - What option did we choose?
   - What happened? (progression, shot, turnover)
   - What was the position score 5 seconds later?
2. Build feedback loop:
   - Engine predicts: "Pass A has 0.7 expected value"
   - Reality shows: "Pass A worked 40% of the time in similar situations"
   - Engine adjusts: Success probability for Pass A in this context is 0.4, not 0.7
3. Over time, physics predictions converge toward reality

**Output:** Engine recommendations based on what works for Marshall, not just what physics says.

**Data requirement:** Minimum 20 labeled matches to start seeing patterns. 50+ for reliable learning.

---

### Phase 5: Player-Level Analysis

**Goal:** Individual feedback, not just team patterns.

**Work:**
1. Track which player made each decision
2. Aggregate: "Player X chose Option A in Situation Y — success rate vs team average"
3. Identify patterns:
   - "Player X consistently picks the second-best option in counter-press"
   - "Player Y finds options others miss in BGZ buildup"
4. Generate player-specific reports

**Output:** Coaching staff gets individual development insights backed by data.

---

## Technical Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      VIDEO INPUT                            │
│              (Broadcast, tactical cam, etc.)                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    TRACKING LAYER                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   YOLO      │  │  ByteTrack  │  │    Homography       │  │
│  │  Detection  │──│   Tracking  │──│   Calibration       │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
              Player positions (x, y, velocity, team)
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   DECISION ENGINE                           │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              ELIMINATION CALCULATOR                    │  │
│  │   For each defender: can they reach intervention      │  │
│  │   point before attacker? Time comparison.             │  │
│  └───────────────────────────────────────────────────────┘  │
│                            │                                │
│                            ▼                                │
│  ┌───────────────────────────────────────────────────────┐  │
│  │            GAME STATE EVALUATOR                        │  │
│  │   Position score (0.0-1.0) based on:                  │  │
│  │   - Elimination ratio (25%)                           │  │
│  │   - Proximity to goal (20%)                           │  │
│  │   - Shooting angle (15%)                              │  │
│  │   - Defensive density (15%)                           │  │
│  │   - Defensive compactness (10%)                       │  │
│  │   - Available actions (15%)                           │  │
│  └───────────────────────────────────────────────────────┘  │
│                            │                                │
│                            ▼                                │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              OPTION EVALUATOR                          │  │
│  │   For each available action (pass, dribble, shot):    │  │
│  │   - Success probability                               │  │
│  │   - Value if successful (position score after)        │  │
│  │   - Risk if failed (position score if turnover)       │  │
│  │   - Expected value = (prob × value) - ((1-prob) × risk)│  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   GAME MODEL LAYER                          │
│   Labels moment type (BGZ, counter-press, transition)       │
│   Adjusts weights based on tactical context                 │
│   Connects decisions to outcomes for learning               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      OUTPUTS                                │
│   - Per-moment analysis (options, recommendations)          │
│   - Match summaries (pattern identification)                │
│   - Player reports (individual decision quality)            │
│   - Training clips (moments flagged for review)             │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Metrics

**Tracking accuracy:**
- Position error < 2m average
- Identity maintained through occlusions 90%+
- Ball detection in traffic 80%+

**Engine validation:**
- Coach agreement on flagged moments: 80%+
- False positive rate (engine flags non-events): < 20%

**Outcome learning:**
- Prediction accuracy improves over time
- Success probability estimates converge toward actual rates

---

## Timeline (No Dates — Milestones Only)

| Milestone | Depends On | Success Criteria |
|-----------|-----------|------------------|
| Tracking produces reliable coordinates | Calibration working | Position error < 2m validated |
| Pipe connected | Reliable tracking | Match produces frame-by-frame analysis JSON |
| Face validity confirmed | Pipe connected | 80% coach agreement on flagged moments |
| Game model integrated | Face validity | Engine identifies moment types correctly |
| Outcome learning active | 20+ labeled matches | Prediction accuracy improving quarter-over-quarter |
| Player-level reports | Outcome learning | Individual decision quality metrics available |

---

## Risks

| Risk | Mitigation |
|------|-----------|
| Tracking never becomes reliable enough | Simplify to set-piece analysis only (static frames) |
| Coaches don't trust engine output | Heavy involvement in validation phase, adjust until it matches intuition |
| Not enough labeled data for learning | Start with opponent analysis (simpler, less data needed) |
| Performance too slow for real-time | Batch processing post-match is acceptable initially |

---

## What This Enables

**For Marshall (immediate):**
- Objective measurement of game model execution
- Player-level feedback on decision quality
- Opponent analysis — model their defensive patterns, find weaknesses

**For any team (general):**
- Plug in different game model definitions
- Same engine, different tactical philosophy
- Outcome learning specific to that team's players and style

The physics are universal. The learning is team-specific.
