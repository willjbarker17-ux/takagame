# Marshall Men's Soccer - Program Intelligence Platform

## Summary

A two-part system that grows with the program:

**Part 1: Marshall-Specific AI** — A knowledge system built on our data. Captures and connects meeting discussions with existing data (Wyscout, GPS, recruitment). Delivers value immediately.

**Part 2: Tracking & Decision Engine** — Computer vision to extract tactical data from match video. Develops in background, eventually feeds into Part 1.

**Key insight:** Part 1 works now. Part 2 makes it better over time.

---

# Part 1: Marshall-Specific AI

## The Core Idea

A system that connects all our program data and makes it searchable and queryable.

Not a chatbot. Not a replacement for Wyscout. **It's the connective tissue between everything** — letting us ask questions across systems that don't talk to each other today.

---

## What Data It Connects

| Source | What It Contains |
|--------|------------------|
| Wyscout | Match events, opponent tendencies, player stats |
| GPS | Training load, distances, sprints, fatigue patterns |
| Game Plans | Tactical documents, scouting reports |
| Meeting Recordings | Discussions, decisions, context (transcribed + video) |
| Recruitment | Prospect notes, evaluations, portal tracking |
| Financials | Budget, scholarships, travel costs |
| Medical | Injury history, return-to-play |
| Schedule | Matches, travel, recovery windows |

**We already have this data.** It's just scattered across systems.

---

## Why This Matters

### 1. Cross-Source Pattern Recognition

Right now, our data lives in separate systems. Wyscout doesn't know about GPS load. GPS doesn't know about meeting discussions. Recruitment notes don't connect to player outcomes.

The system links everything together. When preparing for an opponent, we see not just their Wyscout tendencies but what staff discussed about them last time, how our players performed physically against similar teams, and what adjustments worked.

When evaluating a player's development, we see their GPS trends alongside the development goals set in meetings months ago.

The value is **surfacing the right context at the right time** — connections we wouldn't manually make across separate systems.

---

### 2. Institutional Memory & Staff Alignment

This is the biggest value.

Every program accumulates knowledge through discussions — why certain decisions were made, what concerns existed about players before signing, what tactical adjustments worked against specific opponents, how the coaching philosophy has evolved.

Right now, that knowledge lives in people's heads. Not everyone was in every meeting. When revisiting a decision from last season, we're starting from scratch.

The system preserves it. Every meeting discussion becomes searchable. The reasoning behind decisions is captured, not just the outcomes. Any staff member can access the full context — what was discussed, what was decided, what the thinking was. Everyone works from the same information. Past lessons don't get forgotten and repeated.

---

### 3. Recruitment Intelligence

Recruitment decisions happen through discussions — staff watch film, share observations, debate concerns, reach conclusions. Those discussions contain valuable evaluation signals, but they're never connected back to outcomes.

The system links what we said about prospects to how they actually performed after signing. Over time, we learn which evaluation signals mattered and which didn't. We can identify what successful players in our system had in common during the recruiting process.

This turns recruitment from isolated decisions into a learning system.

---

### 4. Proactive Alerts

The system watches our data and flags things without anyone asking.

Instead of manually checking load reports, scouting updates, or portal activity, the system monitors patterns and surfaces what matters: a player's load trending toward injury risk, an opponent changing their set piece routines since we last scouted them, a recruit entering the portal who matches what we're looking for.

We don't have to remember to check. It tells us.

---

### 5. Auto-Generated Prep

Before a match, the system pulls together everything relevant into one place: opponent tendencies from Wyscout, what staff discussed about them last time we played, how the squad is physically, what adjustments worked historically against similar teams.

Currently this is manual assembly from multiple sources. The system does it automatically, so prep starts from context instead of from scratch.

---

## Timeline

| Phase | Timeframe | What We Get |
|-------|-----------|--------------|
| Data Ingestion | Week 1 | All data connected and indexed |
| Core System | Week 2 | Can query across all data sources |
| Alerts & Prep | Week 3 | Proactive alerts, auto-generated prep docs |

**Week 2: Staff can start querying. Week 3: Full system working.**

---

# Part 2: Tracking & Decision Engine

## The Vision

Most college programs rely entirely on Wyscout for analytics. That's a baseline — everyone has it.

What separates elite programs analytically is **proprietary data** — metrics and insights competitors can't access because they don't have the system to generate them.

This builds that system. Computer vision tracking from our match video, generating data that goes beyond what Wyscout provides, which can then be fed into Part 1's intelligence system.

This is how we can have an analytical edge over everyone else in the country.

---

## Where We Are Now

**Over 36,000 lines of code already written.** The foundation is built:

| Component | Status |
|-----------|--------|
| Player detection | Working |
| Ball detection | Working |
| Manual pitch calibration | Working |
| Basic tracking | Working |
| Physical metrics from video | Working |

What needs work:
- **Automatic pitch calibration** — code exists, needs training data to work reliably
- **Complex scenario tracking** — occlusions, camera cuts, crowded situations
- **Validation** — verify our numbers match GPS ground truth

---

## What Tracking Enables

**Immediate value once tracking is solid:**
- Physical metrics from video when GPS isn't available (opponent analysis)
- Formation shape and spacing analysis
- Pressing triggers and defensive line positions
- Continuous spatial data between events — what Wyscout doesn't capture

Wyscout gives us events (passes, shots, fouls). Tracking gives us the full picture of what happens between those events.

---

## The Game Decision Engine

This is the ambitious part — and what makes the whole system unique. The code framework already exists. Over 36,000 lines written, with the core modules built.

### The Core Concept: Elimination & High xG Zones

The engine is built around two fundamental ideas:

**1. Elimination as tactical currency**

A defender is "eliminated" when:
1. The ball is past them (positionally, toward goal)
2. They can't reach an intervention point before the attacker reaches a shooting point.

This isn't about who's ball-side or goal-side on paper. It's about who can actually affect the play. A defender who's technically goal-side but can't intervene in time is eliminated.

**Every attacking action can be evaluated by how many defenders it eliminates.** A pass that takes out three defenders is more valuable than one that takes out one. The engine quantifies this frame-by-frame.

**2. Getting into high xG zones**

Elimination matters because of where it leads. The goal is getting the ball into positions where expected goals (xG) is highest — central areas close to goal with time and angle to shoot.

The engine connects these: How many defenders did we eliminate to get there? What was the path? Which actions created the dangerous position?

This lets us measure attacking quality beyond just "did we score" — we can see whether we're consistently creating high-value chances and how we're getting there.

### Football as Physics

The engine models football as a dynamic spatial system governed by forces:

**Defensive positioning** is modeled through attraction forces:
| Force | Effect |
|-------|--------|
| Ball attraction | Creates pressing behavior |
| Goal attraction | Creates protective depth |
| Opponent attraction | Creates marking |
| Teammate repulsion | Maintains spacing |
| Line attraction | Maintains compactness |

The equilibrium of these forces determines where defenders should be. When they're not there, we can measure the gap between actual and optimal.

**Game state scoring** evaluates every moment:
| Component | What It Measures |
|-----------|------------------|
| Elimination | Defenders taken out of the play |
| Proximity | Distance to goal |
| Angle | Shooting angle available |
| Density | Space around the ball |
| Compactness | Defensive structure integrity |
| Action availability | Forward passing options |


### The Philosophy: Structure Before Talent

The base model treats all players as physically equal — same speed, same reaction time. This is intentional.

**We understand structure before we layer in talent.** If our positioning fails when everyone's equal, it will fail when talent is added. Talent becomes a modifier on top of sound structure, not a replacement for it.

This is how we identify whether breakdowns are positional (fixable through coaching) or physical (need different personnel).

### The Pipeline

```
Video → Detection → Tracking → Coordinates → Decision Engine → Analysis
        (YOLO)    (ByteTrack) (Calibration)   (Elimination,    (Insights)
                                               Forces, Scoring)
```

The decision engine sits at the end of the tracking pipeline. It takes coordinate data and produces tactical analysis.

### What It Enables

**Post-match:** The engine flags specific moments where structure failed — which defenders got eliminated, where the block broke, what triggered it.

**Trends over time:** We build a dataset across the season. Is our compactness improving? Are we eliminating more defenders per attack? Are there fatigue patterns where execution drops?

**Opponent scouting:** Apply the same engine to opponent film. Model their defensive block, find where they're vulnerable, identify what triggers their breakdowns.

### Two Modes of Operation

**1. Standalone Simulation Mode**

The engine can run without any video input. We define a scenario — place players on the pitch, set a ball position — and the engine calculates:

- Highest xg available actions (pass options, dribble lanes)
- What the optimal defensive positioning should be (based on the force model)
- Game state score for the attacking team

The engine becomes a tactical sandbox. We can test ideas computationally and see the best way to win

**2. Real Match Analysis Mode**

Once tracking is validated, we feed actual match coordinates into the engine. Now we're not simulating — we're measuring what actually happened.

For any moment in a match:
- Did our player choose the highest xg option?
- Where was the opponent's most effective point of attack?
- Did we get into high xG zones? What path did we take?


**How They Connect**

The standalone mode establishes baselines and targets. The real match mode measures against them.

Example: In simulation, we determine that against a 4-1-4-1 press, this is the best pattern to find a free 10. In real matches, the engine measures whether we achieved that. The gap between target and reality becomes the coaching focus.


### Current State

The core modules are built:
- `elimination.py` — Core elimination logic
- `defense_physics.py` — Attraction-based positioning model
- `state_scoring.py` — Game state evaluation
- `block_models.py` — Defensive block configurations
- `visualizer.py` — Tactical board output

What remains is **calibration against real match data** — running it on our film, refining thresholds, confirming it flags the moments coaches identify.

### Why This Takes Time

The tracking must be accurate first. If player positions are wrong, every elimination calculation is garbage. That's why tracking validation comes before decision engine deployment.

Then the parameters need tuning. How much should ball attraction outweigh goal attraction? What's the threshold for "eliminated"? These require iteration with coaching staff watching outputs and refining.

**Realistic expectation:** Tracking delivers value by spring. Decision engine v1 by fall — starting with elimination metrics and block analysis, then expanding as we validate.

---

## Development Timeline

| Phase | Timeframe | Outcome |
|-------|-----------|---------|
| Calibration training | Month 1 | Process any video without manual setup |
| Tracking refinement | Month 2 | Handle real match conditions reliably |
| Validation against GPS | Month 3 | Confirm accuracy, ready for use |
| Decision engine v1 | Months 4-5 | First Game Model metrics automated |

**Spring:** Tracking integrated with Part 1 — we can ask questions that combine Wyscout events with our spatial data.

**Fall:** Decision engine measuring tactical principles. This is where it gets powerful.

---

# How They Work Together

```
Part 1 (Marshall AI) - Working in Weeks
───────────────────────────────────────
Week 1:     Data ingestion (Wyscout, GPS, meetings)
Week 2:     Core system working         ← Value starts here
Week 3:     Alerts + auto-prep


Part 2 (Tracking) - Developing in Parallel
───────────────────────────────────────
Month 1-2:  Calibration + tracking refinement
Month 3:    Validation, integration     ← Feeds into Part 1
Month 4-5:  Decision engine v1          ← Game Model automation
```

Part 1 delivers value in weeks. Part 2 compounds it over months — and creates the analytical edge no one else has.

---

# Summary

**Part 1:** Connect our scattered data (Wyscout, GPS, meetings, recruitment) into one queryable system. Surface patterns across sources, preserve institutional knowledge, track recruitment outcomes, get proactive alerts, and auto-generate match prep. Working in 2-3 weeks.

**Part 2:** Computer vision tracking that goes beyond Wyscout — proprietary spatial data, then automated Game Model measurement. Over 36,000 lines already written. Tracking ready by spring, decision engine by fall.

**The key insight:** Everyone has Wyscout. This system connects our scattered data, then builds proprietary analytics on top. That's how we become more analytical than everyone else in the country.
