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
| Data Ingestion | Week 1 | Highest-signal sources connected (docs, meetings, Wyscout) |
| Core System | Week 2 | Working Hub, staff can start querying |
| Expansion | Week 3+ | GPS, recruitment, continuous improvement |

The point is adoption and compounding, not claiming perfection on Day 21.

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

Tracking produces coordinates. The engine turns those coordinates into objective tactical information.

It's built around three measurable primitives:

**1. Elimination**
Not "goal-side" in theory — who can actually affect the play in time. A defender who looks close enough but can't intervene before the attacker progresses is eliminated. This quantifies what coaches see intuitively.

**2. Structure Integrity**
Compactness, line spacing, block stability, xG path coverage, and the precise moments structure breaks. Includes a defensive model with adjustable parameters — we can tune it to match how opponents actually defend, or measure ourselves against our own targets.

**3. Game-State Scoring**
A consistent score for any moment based on elimination, proximity, angle, density, compactness, and action availability. This lets us compare situations objectively, not by vibe.

This is how we turn "I think we should've switched it" into "switching to the weak side here was +0.12 xG; the pass to the winger was +0.04 xG; holding possession was +0.02 xG." We can see which decisions actually created value and which just felt right.

The core modules are built: `elimination.py`, `defense_physics.py`, `state_scoring.py`, `block_models.py`, `visualizer.py`. What remains is calibration against real match data.

---

## What It Enables

**Decision clarity:** For any moment in a match, see what the highest-value action was and compare it to what we actually did. Not "we should've done better" — specific alternatives with specific xG values attached.

**Simulating alternatives:** After a goal conceded, test what positioning would have prevented it. Before a match, simulate responses to opponent shapes. "If we overload the left against their 4-4-2, where do they become vulnerable?"

**Opponent scouting:** Model their defensive tendencies mathematically, find where they break down. Know their weaknesses before kickoff, not after watching them exploit ours.

**Trends over time:** Track whether we're getting more compact, eliminating more defenders, where fatigue affects execution. See if what we're training is actually showing up in matches.

---

## Sequenced Scope

| Stage | Focus | Outcome |
|-------|-------|---------|
| A | Tracking outputs | Shape, spacing, baseline structure metrics |
| B | Decision Engine v1 | Elimination + compactness/block measurements |
| C | Engine expansion | Richer scoring, scenario simulation |

We do not pretend the full engine is useful before tracking is trustworthy.

---

## Timeline

| Phase | Timeframe | Outcome |
|-------|-----------|---------|
| Calibration training | Month 1 | Process video without manual setup |
| Tracking refinement | Month 2 | Handle real match conditions |
| Validation | Month 3 | Confirm accuracy against GPS |
| Decision Engine v1 | Months 4-5 | First game model metrics automated |

**Spring:** Tracking integrated with Hub — query across Wyscout events and our spatial data.

**Fall:** Decision engine measuring tactical principles. This is where it gets powerful.

---

## Definition of Done

Part 2 is "real" when:
- Calibration is consistent without heroic manual setup
- Tracking accuracy is validated against GPS where possible
- The engine flags the same moments coaches identify on film (face validity)
- Outputs are stable enough to compare across matches (repeatability)

---

# How They Work Together

```
Part 1 (Hub) - Working in Weeks
───────────────────────────────
Week 1:     Data ingestion
Week 2:     Core system working    ← Value starts here
Week 3+:    Expansion, adoption


Part 2 (Tracking + Engine) - Developing in Parallel
───────────────────────────────────────────────────
Month 1-2:  Calibration + tracking
Month 3:    Validation             ← Feeds into Hub
Month 4-5:  Decision engine v1     ← Game model automation
```

Part 1 makes everything searchable and connected. Part 2 adds a new class of proprietary evidence inside that system — continuous tactical reality and objective measurements of our principles.

The Hub isn't just "organized information." It becomes a system that can learn and surface truths from our games in ways competitors can't reproduce.

---

# Summary

**Part 1:** Connect our scattered data into one queryable system. Staff alignment becomes structural. Pattern recognition across sources we can't mentally integrate. Institutional knowledge compounds instead of resetting. Working in 2-3 weeks.

**Part 2:** Proprietary tracking + decision engine that measures our game model objectively. 36,000+ lines already written. Tracking by spring, decision engine by fall.

**The key insight:** Everyone has Wyscout. This system connects our scattered data, then builds proprietary analytics on top. That's how we become more analytical than everyone else in the country.

---

*For technical details on the full system architecture, see: TECHNICAL_BRIEF.md*
