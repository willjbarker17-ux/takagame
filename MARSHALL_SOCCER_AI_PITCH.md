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

### 2. Institutional Memory & Staff Alignment

This is the biggest value.

Every program accumulates knowledge through discussions — why certain decisions were made, what concerns existed about players before signing, what tactical adjustments worked against specific opponents, how the coaching philosophy has evolved.

Right now, that knowledge lives in people's heads. Not everyone was in every meeting. When you revisit a decision from last season, you're starting from scratch.

The system preserves it. Every meeting discussion becomes searchable. The reasoning behind decisions is captured, not just the outcomes. Any staff member can access the full context — what was discussed, what was decided, what the thinking was. Everyone works from the same information. Past lessons don't get forgotten and repeated.

---

### 3. Recruitment Intelligence

Recruitment decisions happen through discussions — staff watch film, share observations, debate concerns, reach conclusions. Those discussions contain valuable evaluation signals, but they're never connected back to outcomes.

The system links what you said about prospects to how they actually performed after signing. Over time, you learn which evaluation signals mattered and which didn't. You can identify what successful players in your system had in common during the recruiting process.

This turns recruitment from isolated decisions into a learning system.

---

### 4. Proactive Alerts

The system watches your data and flags things without you asking.

Instead of manually checking load reports, scouting updates, or portal activity, the system monitors patterns and surfaces what matters: a player's load trending toward injury risk, an opponent changing their set piece routines since you last scouted them, a recruit entering the portal who matches what you're looking for.

You don't have to remember to check. It tells you.

---

### 5. Auto-Generated Prep

Before a match, the system pulls together everything relevant into one place: opponent tendencies from Wyscout, what your staff discussed about them last time you played, how your squad is physically, what adjustments worked historically against similar teams.

Currently this is manual assembly from multiple sources. The system does it automatically, so prep starts from context instead of from scratch.

---

## Timeline

| Phase | Timeframe | What You Get |
|-------|-----------|--------------|
| Data Ingestion | Week 1 | All data connected and indexed |
| Core System | Week 2 | Can query across all data sources |
| Alerts & Prep | Week 3 | Proactive alerts, auto-generated prep docs |

**Week 2: Staff can start querying. Week 3: Full system working.**

---

# Part 2: Tracking & Decision Engine

## The Vision

Most college programs rely entirely on Wyscout for match data. That's a baseline — everyone has it.

What separates elite programs analytically is **proprietary data** — metrics and insights competitors can't access because they don't have the system to generate them.

Part 2 builds that system. Computer vision tracking from match video, generating data that goes beyond what Wyscout provides, then feeding it into Part 1's intelligence system.

This is how we can become more analytical than everyone else in the country.

---

## Where We Are Now

The foundation is built:

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
- Possession statistics, field tilt, territorial control
- Formation shape and spacing analysis
- Pressing triggers and defensive line positions

**This fills the gap Wyscout can't** — Wyscout gives you events (passes, shots, fouls) but not the continuous spatial data between events. Tracking gives you the full picture.

---

## The Game Decision Engine

This is the ambitious part.

Once we have reliable tracking data, we can build toward measuring your Game Model principles automatically. Did we win the second ball? How quickly did we transition? Were we compact in our defensive shape?

The code framework exists. But making it accurate requires:
1. Validated tracking data (no garbage in, garbage out)
2. Precise definitions of each principle
3. Testing against film to confirm it matches what coaches see

**Realistic expectation:** Tracking delivers value by spring. Decision engine is a fall project — the kind of thing that could genuinely change how you prepare for matches, but needs time to get right.

---

## Development Timeline

| Phase | Timeframe | Outcome |
|-------|-----------|---------|
| Calibration training | Month 1 | Process any video without manual setup |
| Tracking refinement | Month 2 | Handle real match conditions reliably |
| Validation against GPS | Month 3 | Confirm accuracy, ready for use |
| Decision engine v1 | Months 4-5 | First Game Model metrics automated |

**Spring:** Tracking integrated with Part 1 — you can ask questions that combine Wyscout events with our spatial data.

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

# What We Need

### Data Access
- Wyscout export or API access
- GPS data exports
- Game plans and scouting documents
- Recruitment tracking
- Permission to record and transcribe meetings

---

# Summary

**Part 1:** Connect your scattered data (Wyscout, GPS, meetings, recruitment) into one queryable system. Surface patterns across sources, preserve institutional knowledge, track recruitment outcomes, get proactive alerts, and auto-generate match prep. Working in 2-3 weeks.

**Part 2:** Computer vision tracking that goes beyond Wyscout — proprietary spatial data, then automated Game Model measurement. Tracking ready by spring, decision engine by fall.

**The key insight:** Everyone in your conference has Wyscout. This system connects your scattered data, then builds proprietary analytics on top. That's how you become more analytical than everyone else.
