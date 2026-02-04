# Marshall Men's Soccer – Program Intelligence Platform

## The idea in one line

A two-part system that aligns us operationally now and builds a proprietary tactical edge over time.

---

**Part 1: Marshall Intelligence Hub** — A Marshall-specific knowledge system that connects our existing data and discussions into one searchable, queryable program brain. Delivers value in weeks.

**Part 2: Tracking + Decision Engine** — Computer vision generates proprietary spatial data from match video; a decision engine converts that data into objective measurements of our game model. Develops over months.

**Key point:** Part 1 works now. Part 2 upgrades it into something no one else in college soccer has.

---

# Part 1: Marshall Intelligence Hub

## What It Is

Not a chatbot. Not a Wyscout replacement.

It's the connective tissue across everything we already produce — data, documents, and conversations — so the program operates from one shared base of truth.

## What It Connects

| Source | What It Contains |
|--------|------------------|
| Wyscout | Match events, opponent tendencies, player stats |
| GPS | 240+ metrics per session — load, distances, sprints, fatigue patterns |
| Game Plans | Tactical documents, scouting reports |
| Meeting Recordings | Discussions, decisions, context (transcribed) |
| Recruitment | Prospect notes, evaluations, portal tracking |
| Financials | Budget, scholarships, travel costs |
| Medical | Injury history, return-to-play |
| Schedule | Matches, travel, recovery windows |

**We already have this data.** It's just scattered across systems that don't talk to each other.

---

## Why It Matters

### 1. Staff Learns Together, Not Alone

Right now, alignment relies on memory, repetition, and who was in which room. The Hub makes alignment persistent:

- The logic behind decisions is captured
- The context stays attached to outcomes
- Discussions don't evaporate after the week ends

But it's more than just storage. The system collects all of our inputs — observations, evaluations, tactical discussions — and learns more about how Marshall thinks. Over time, it gets better at surfacing what matters to us specifically, not generic insights. The more we use it, the more it understands our program's way of seeing the game.

We stop constantly re-deriving what we believe, and start building on what we've already figured out.

### 2. Pattern Recognition Across Systems We Can't Mentally Integrate

This is the real advantage.

We already have meaningful signals in separate places — Wyscout, GPS, notes, meetings — but no human can reliably combine them across time (months/years), modalities (numbers + video + language), and volume (every session, every match, every discussion).

The Hub creates value by linking and surfacing relationships we can't comprehend manually. Not "more data." More intelligence from the data we already have.

We stop treating information as isolated events and start treating it as a connected system that reveals:
- Recurring themes in what we say vs what happens
- Trends that only appear when multiple sources are viewed together
- What actually correlates with outcomes in our environment

### 3. Institutional Knowledge Becomes Cumulative

The program should compound. Right now, it resets more than it should because reasoning and context aren't stored in a way we can query later.

The Hub turns our weekly thinking into an asset we can leverage repeatedly, rather than re-create.

---

## What Success Looks Like

- We move faster because we start from context, not from scratch
- We're aligned because decisions and rationale are visible and consistent
- We get smarter over time because the system learns across seasons of our own behavior and outcomes

---

## Timeline

| Phase | Timeframe | What We Get |
|-------|-----------|-------------|
| Data Ingestion | Weeks 1-2 | Highest-signal sources connected (docs, meetings, Wyscout) |
| Core System | Weeks 3-4 | Working Hub, staff can start querying |
| Refinement | Weeks 5-6 | Iterate based on staff feedback, improve retrieval quality |
| Expansion | Ongoing | GPS, recruitment, continuous improvement |

The point is adoption and compounding, not claiming perfection on Day 30.

---

# Part 2: Tracking + Decision Engine

## Why Part 2 Exists

Everyone has baseline analytics. If we want a real analytical edge, we need proprietary data and a proprietary model.

Wyscout is events. Tracking is continuous reality: shape, spacing, timing, and what happens between events.

But the real differentiator isn't just tracking — it's what we do with it.

Part 2 is two layers:
1. **Tracking Layer** (data creation): coordinates from video
2. **Decision Engine Layer** (interpretation): converts coordinates into game-model measurements

---

## Layer 1: Tracking

The goal is extracting player and ball coordinates from Wyscout match video to generate spatial data we don't currently have.

**Current state (honest):** Over 36,000 lines of code written. We can detect players and the ball in individual frames. But reliable tracking across the full field — maintaining identity through occlusions, camera cuts, and crowded areas — isn't there yet. Same with the ball in traffic. Pitch calibration works when the camera angle is stable, but Wyscout's wide-angle footage shifts, and recalibrating on the fly is the main technical challenge.

**What needs to happen:**
- Automatic pitch calibration that handles camera angle changes
- Tracking that maintains player identity across the full 90 minutes
- Validation against GPS to confirm accuracy

Tracking is only useful if it's accurate under real match conditions. That's why validation is the gate before we build anything on top of it.

---

## Layer 2: Decision Engine

Tracking produces coordinates. The engine turns those coordinates into tactical intelligence.

**The End Goal: A Football Simulation Engine**

The vision is an engine that can simulate any game situation and find the optimal path to a scoring opportunity — not through designed rules, but by learning from thousands of real match situations what actually works.

Given a position, explore possible action sequences, prune bad options, go deep on promising ones, and return the best path forward.

**The Approach: Learn From Reality, Then Simulate**

We're not designing simulation rules that might be wrong. We're learning how football actually works from tracking data:

```
Phase 1-4: Learn from real matches
           ↓
           "In situations like this, passes to the wing succeed 65% of time"
           "Defenders typically react by shifting this direction"
           "This type of dribble works against high pressure"
           ↓
Phase 5:   Use learned patterns as simulation rules
           ↓
           Search for optimal sequences through simulation
```

**What we're building first:**

A similarity-based system that:
1. Encodes each game situation as features (defensive shape, ball position, pressure level)
2. Finds similar situations from our database of real matches
3. Analyzes what actions succeeded in those situations
4. Learns the actual success rates — not physics guesses, real outcomes

**What it can do now:**

There's a working interactive tactical board with physics-based analysis:
- Calculates xG from any position
- Evaluates pass, dribble, and shot options
- Finds gaps in defensive lines
- Detects through ball opportunities
- Ranks options by expected value

The physics are calibrated to real data: sprint acceleration, top speed, reaction time, ball deceleration. This gives us a foundation.

**What comes next:**

Connect to real game footage. Feed it tracking data from actual matches. Build a database of situations and outcomes. The engine learns: "Physics says this pass is available, but in similar situations it actually worked 40% of the time."

**The simulation goal (Phase 5):**

Once we've learned enough patterns, we can run simulations:

```
Ball reception moment
    ├── Pass to Player A
    │      ├── A shoots → evaluate xG
    │      ├── A passes to B → keep exploring
    │      └── A loses ball → stop
    ├── Pass to Player B (promising) → go DEEP
    ├── Dribble into pressure → PRUNE early
    └── Shot → evaluate xG
```

Variable depth. Prune bad branches. Go deep on promising ones. Find the path with highest expected value of reaching a scoring position.

Start by "solving" at ball reception moments. As the system improves, solve more frequently — eventually analyzing the game near-continuously.

---

## What It Enables

**Decision clarity:** For any moment in a match, see what the highest-value action was and compare it to what we actually did. Not "we should've done better" — specific alternatives with specific xG values attached.

**Simulating alternatives:** After a chance is conceded, test what positioning would have prevented it. Before a match, simulate our structures against their shape.

**Opponent scouting:** Build defensive and attacking profiles for any team. Identify their pressing triggers, recovery patterns, where they get stretched.

**Trends over time:** Track whether we're executing our principles across the season. See if what we're training is actually showing up in matches.

**Recruitment evaluation:** Run a prospect's film through the engine. Extract physical metrics from video — sprint patterns, recovery runs, work rate — without needing their GPS data. Analyze their decision-making objectively.

---

## Timeline

| Phase | Timeframe | Outcome |
|-------|-----------|---------|
| Calibration training | Month 1 | Process video without manual setup |
| Tracking refinement | Month 2 | Handle real match conditions |
| Validation | Month 3 | Confirm accuracy against GPS |
| Decision Engine v1 | Months 4-5 | First game model metrics automated |

**Spring:** Tracking integrated with Hub — query across Wyscout events and our spatial data.

**Fall:** Decision engine measuring tactical principles.

---

# How They Work Together

```
Part 1 (Hub) - Working in Weeks
───────────────────────────────
Weeks 1-2:  Data ingestion
Weeks 3-4:  Core system working    ← Value starts here
Weeks 5-6:  Refinement based on feedback
Ongoing:    Expansion, adoption


Part 2 (Tracking + Engine) - Developing in Parallel
───────────────────────────────────────────────────
Month 1-2:  Calibration + tracking
Month 3:    Validation             ← Feeds into Hub
Month 4-5:  Decision engine v1     ← Game model automation
```

Part 1 makes everything searchable and connected. Part 2 adds a new class of proprietary evidence inside that system — continuous tactical reality and objective measurements of our principles.

---

# Summary

**Part 1:** Connect our scattered data into one queryable system. Staff alignment becomes structural. Pattern recognition across sources we can't mentally integrate. Institutional knowledge compounds instead of resetting. Working in 4-6 weeks.

**Part 2:** Proprietary tracking + decision engine. The end goal: a football simulation engine that finds optimal paths to scoring positions by learning from real match data. 36,000+ lines already written. Tracking by spring, decision engine by fall.

**The key insight:** Everyone has Wyscout. This system connects our scattered data, learns what actually works from our games, then builds a simulation grounded in reality. That's how we become more analytical than everyone else in the country.
