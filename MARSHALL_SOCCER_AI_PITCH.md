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

Most college programs rely entirely on Wyscout for match data. That's a baseline — everyone has it.

What separates elite programs analytically is **proprietary data** — metrics and insights competitors can't access because they don't have the system to generate them.

Part 2 builds that system. Computer vision tracking from our match video, generating data that goes beyond what Wyscout provides, then feeding it into Part 1's intelligence system.

This is how we become more analytical than everyone else in the country.

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

This is the ambitious part — and what makes the whole system unique.

### What It Does

The decision engine takes our tracking data and measures whether we're executing our Game Model principles. Not subjective film review — objective, frame-by-frame measurement.

**Examples of what it can measure:**
- **Transition speed:** How many seconds from winning possession to penetrating their half? How does that compare to our target?
- **Defensive compactness:** Distance between our lines when defending. Are we staying connected or getting stretched?
- **Pressing triggers:** When we lose the ball, how quickly do we engage? Who's pressing and who's covering?
- **Second ball wins:** After long balls or aerial duels, are we positioned to collect?
- **Width in possession:** Are our wingers providing proper width? How does spacing change by phase?
- **Recovery runs:** When we lose possession, are players tracking back at the right intensity?

### How It Works

The engine builds on two layers:

1. **Wyscout events** — passes, shots, duels, set pieces. This is the "what happened."
2. **Our tracking data** — continuous player positions between events. This is the "how it happened."

By combining them, we can answer questions Wyscout alone can't: We completed 85% of our passes, but were we playing through the lines or going sideways? We won 60% of aerial duels, but were we positioned to win the second balls?

### Development Process

This isn't plug-and-play. Building it right requires:

1. **Define each principle precisely** — "Compact defensive shape" needs to be translated into measurable terms. What's the maximum distance between lines? What triggers an alert?

2. **Calibrate against film** — Run the engine on matches we've already analyzed manually. Does it flag the same moments coaches identified? If not, refine.

3. **Iterate with feedback** — The first version won't be perfect. We run it, coaches review outputs, we adjust thresholds and definitions.

This is collaborative work between the technical side and coaching staff. The goal is a system that reflects how we actually think about the game.

### What It Enables

**Post-match:** Instead of watching full film to find breakdowns, the engine flags specific moments where principles weren't executed. Review becomes targeted.

**Over time:** We build a dataset of principle execution across the season. We can see trends — are we getting more compact? Is transition speed improving? Are there fatigue patterns where execution drops?

**Opponent analysis:** Apply the same engine to opponent film. Identify their principles, find where they break down, prepare to exploit it.

**Recruitment:** Evaluate prospects not just on Wyscout stats but on whether their playing style fits our principles.

### Why This Takes Time

The tracking must be accurate first. If player positions are wrong, every principle measurement is garbage. That's why tracking validation comes before decision engine development.

Then the definitions need refinement. "Compactness" is intuitive when watching film but requires precise parameters for automated measurement. Getting those parameters right takes iteration with coaching staff.

**Realistic expectation:** Tracking delivers value by spring. Decision engine v1 by fall — and it keeps improving as we refine definitions and add more principles.

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
