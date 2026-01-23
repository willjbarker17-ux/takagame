# Marshall Men's Soccer - Program Intelligence Platform

## Summary

We're proposing a two-part system that grows with the program:

**Part 1: Marshall-Specific AI** — A knowledge system built on Marshall data. The key is capturing and connecting meeting discussions with your existing data (Wyscout, GPS, recruitment). Delivers value immediately.

**Part 2: Tracking & Decision Engine** — Computer vision to extract tactical data from match video. Develops in background, eventually feeds into Part 1.

**Key insight:** Part 1 works now. Part 2 makes it better over time.

---

# Part 1: Marshall-Specific AI

## The Core Idea

Meetings are where the real knowledge lives — the reasoning behind decisions, the context around players, the adjustments discussed but never written down.

Right now that knowledge disappears. It lives in memory, scattered notes, or nowhere.

**The system captures meetings (video + audio + transcripts) and connects them to your structured data (Wyscout, GPS, recruitment).** This creates something none of your current tools can provide: searchable institutional memory with context.

---

## What Data It Connects

| Source | What It Contains |
|--------|------------------|
| Meeting Recordings | Discussions, decisions, reasoning, clips shown, context |
| Wyscout | Match events, opponent tendencies, player stats |
| GPS | Training load, distances, sprints |
| Game Plans | Tactical documents, scouting reports |
| Recruitment | Prospect notes, evaluations |
| Medical | Injury history |

---

## Why This Matters

### 1. Cross-Source Pattern Recognition

Connecting meetings to data lets you find patterns:

- "What did we discuss about opponents who press high? Pull up those meetings alongside our results against them."
- "Show me every time we talked about Player X's development — what did we say, and how do his GPS numbers compare now?"
- "What load patterns came before injuries? What did we say about those players before they got hurt?"

The value isn't answering complex tactical questions automatically — it's **surfacing the right context** when you need it.

---

### 2. Institutional Memory

This is the biggest value. Knowledge that currently gets lost:

**Decisions and reasoning:**
- "Why did we change our pressing triggers against Team Y?"
- "What were the concerns about Recruit Z before we signed him? Were they valid?"
- "What adjustments did we discuss at halftime in the games we were losing?"

**Player context:**
- "What development priorities did we set for Player X in September?"
- "When did we first talk about moving Player Y to a new position?"

**Opponent history:**
- "What did we say about Team X last time we played them? What worked?"
- "Show me the meeting where we broke down their set pieces"

**System explanations:**
- "When a new player asks about our defensive transition principles, what's the explanation we've given before?"
- "Play the clip where we explained the difference between overlaps and underlaps"

Without this, every conversation starts from scratch. New staff have no context. Past lessons get forgotten and repeated.

---

### 3. Recruitment Intelligence

Connect recruitment discussions to outcomes:

- "What did we say about recruits similar to Player X before we signed them? How did they turn out?"
- "Pull up all discussions about fullback prospects this year"
- "What concerns came up about Recruit Y's positioning? Show me the meeting clips"
- "Which recruiting channels have produced players who succeeded in our system?"

The system connects what you said about prospects to how they actually performed — so you learn what evaluation signals mattered.

---

## Meeting Capture Setup

**Basic Setup (~$50):**
- HDMI capture dongle between laptop and TV
- Phone for audio backup
- Free transcription (Whisper AI)

**What gets captured:**
- Everything shown on screen (Wyscout clips, tactical diagrams)
- Full audio → transcribed and searchable
- Linked together: "Show me when we discussed Team X" → returns video + transcript

---

## Timeline

| Phase | Timeframe | What You Get |
|-------|-----------|--------------|
| Data Ingestion | Weeks 1-3 | All data connected and indexed |
| Core System | Weeks 4-6 | Can search across meetings + data |
| Refinement | Weeks 7-12 | Improved accuracy, feedback loop |

**Week 6: Staff can start searching meetings and connecting them to data.**

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
Week 1-3:   Data ingestion
Week 4-6:   System working              ← Value starts here
Week 7-12:  Refinement


Part 2 (Tracking) - Background Development
───────────────────────────────────────
Month 1-2:  Automatic calibration
Month 3-4:  Tracking improvements
Month 5-6:  Validation + integration    ← Feeds into Part 1
```

Part 1 delivers value immediately. Part 2 makes it more powerful over time.

---

# What We Need

### Data Access
- Permission to record and transcribe meetings
- Wyscout export or API access
- GPS data exports
- Recruitment tracking
- Game plans and scouting documents

### Equipment (~$50)
- HDMI capture dongle
- Phone tripod

---

# Cost Estimate

| Usage Level | Monthly Cost |
|-------------|--------------|
| Light | $50-75 |
| Moderate | $150-200 |
| Heavy | $300-400 |

Main cost is LLM API usage. Storage is minimal.

---

# Summary

**Part 1:** Capture meetings, connect them to your data, build searchable institutional memory. Value in weeks.

**Part 2:** Tracking and tactical analysis from video. Develops in background over months.

**The key insight:** The most valuable data you have is in your discussions — and right now it disappears. This system keeps it.
