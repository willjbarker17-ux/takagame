# Marshall Men's Soccer - Program Intelligence Platform

## Summary

We're proposing a two-part system that grows with the program:

**Part 1: Marshall-Specific AI** — A knowledge system built on Marshall data. The key is capturing and connecting meeting discussions with your existing data (Wyscout, GPS, recruitment). Delivers value immediately.

**Part 2: Tracking & Decision Engine** — Computer vision to extract tactical data from match video. Develops in background, eventually feeds into Part 1.

**Key insight:** Part 1 works now. Part 2 makes it better over time.

---

# Part 1: Marshall-Specific AI

## The Core Idea

A system that connects all your program data and makes it searchable and queryable.

Not a chatbot. Not a replacement for Wyscout. **It's the connective tissue between everything** — letting you ask questions across systems that don't talk to each other today.

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

**You already have this data.** It's just scattered across systems.

---

## Why This Matters

### 1. Cross-Source Pattern Recognition

Right now, your data lives in separate systems. Wyscout doesn't know about GPS load. GPS doesn't know about meeting discussions. Recruitment notes don't connect to player outcomes.

The system links everything together. When you're preparing for an opponent, you see not just their Wyscout tendencies but what your staff discussed about them last time, how your players performed physically against similar teams, and what adjustments worked.

When you're evaluating a player's development, you see their GPS trends alongside the development goals you set in meetings months ago.

The value is **surfacing the right context at the right time** — connections you wouldn't manually make across separate systems.

---

### 2. Institutional Memory

This is the biggest value.

Every program accumulates knowledge through discussions — why certain decisions were made, what concerns existed about players before signing, what tactical adjustments worked against specific opponents, how the coaching philosophy has evolved.

Right now, that knowledge lives in people's heads. When staff turn over, it's gone. When you revisit a decision from last season, you're starting from scratch.

The system preserves it. Every meeting discussion becomes searchable. The reasoning behind decisions is captured, not just the outcomes. New staff can access years of context instead of starting blind. Past lessons don't get forgotten and repeated.

---

### 3. Recruitment Intelligence

Recruitment decisions happen through discussions — staff watch film, share observations, debate concerns, reach conclusions. Those discussions contain valuable evaluation signals, but they're never connected back to outcomes.

The system links what you said about prospects to how they actually performed after signing. Over time, you learn which evaluation signals mattered and which didn't. You can identify what successful players in your system had in common during the recruiting process.

This turns recruitment from isolated decisions into a learning system.

---

## Timeline

| Phase | Timeframe | What You Get |
|-------|-----------|--------------|
| Data Ingestion | Weeks 1-2 | All data connected and indexed |
| Core System | Weeks 3-4 | Can query across all data sources |
| Refinement | Weeks 5-6 | Improved accuracy, feedback loop |

**Week 4: Staff can start asking questions and getting useful answers.**

---

# Part 2: Tracking & Decision Engine

## What It Is

Computer vision system that extracts player positions and tactical data from match video.

This is longer-term development that eventually feeds richer data into Part 1.

---

## Current Status

| Component | Status |
|-----------|--------|
| Player detection | Working |
| Ball detection | Working |
| Manual pitch calibration | Working |
| Automatic pitch calibration | Code exists, needs training |
| Basic tracking | Working (limitations in complex scenarios) |
| Physical metrics from video | Working |
| Decision engine code | Exists, needs validated tracking data |

---

## What It Will Enable (Future)

Once tracking is reliable:
- Physical metrics from video (when GPS isn't available)
- Possession and field tilt statistics
- Formation shape analysis
- Eventually: Game Model principle measurement

---

## Development Timeline

| Phase | Timeframe | Goal |
|-------|-----------|------|
| Automatic Calibration | Months 1-2 | Process video without manual setup |
| Tracking Improvements | Months 3-4 | Handle occlusions, camera movement |
| Validation | Months 5-6 | Verify accuracy against GPS |
| Integration | Month 6+ | Tracking data flows into Part 1 |

---

# How They Work Together

```
Part 1 (Marshall AI) - Immediate Value
───────────────────────────────────────
Week 1-2:   Data ingestion
Week 3-4:   System working              ← Value starts here
Week 5-6:   Refinement


Part 2 (Tracking) - Background Development
───────────────────────────────────────
Month 1-2:  Automatic calibration
Month 3-4:  Tracking improvements
Month 5-6:  Validation + integration    ← Feeds into Part 1
```

Part 1 delivers value in weeks. Part 2 makes it more powerful over months.

---

# What We Need

### Data Access
- Wyscout export or API access
- GPS data exports
- Game plans and scouting documents
- Recruitment tracking
- Permission to record and transcribe meetings

---

# Summary

**Part 1:** Capture meetings, connect them to your data, build searchable institutional memory. Value in weeks.

**Part 2:** Tracking and tactical analysis from video. Develops in background over months.

**The key insight:** The most valuable data you have is in your discussions — and right now it disappears. This system keeps it.
