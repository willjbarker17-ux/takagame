# Marshall AI - Internal Development Plan

This is our working plan, not the pitch to Marshall.

---

# Two-Part Strategy

## Part 1: Tracking & Decision Engine
**What:** Computer vision system to extract player positions and tactical analysis from video
**Timeline:** Months of development
**Value:** Delayed until working reliably

## Part 2: Marshall-Specific AI
**What:** Knowledge system trained on all Marshall program data
**Timeline:** Can start immediately
**Value:** Instant - data already exists

**Key insight:** Part 2 delivers value NOW while Part 1 develops. Part 1 eventually feeds into Part 2 as another data source.

---

# Part 2: Marshall-Specific AI

## Available Data Sources

| Data Source | What It Contains | Available Now? |
|-------------|------------------|----------------|
| Wyscout | Match events, player stats, opponent tendencies, video | Yes |
| GPS (Catapult/other) | Training load, distances, sprints, accelerations | Yes |
| Financials | Budget, scholarships, travel, staff costs | Yes |
| Game Plans | Tactical documents, formation notes, opponent scouting | Yes |
| Meeting Recordings | Decisions, context, discussions (transcribed) | Yes |
| NCAA Rulebook | Compliance rules, deadlines, limits | Yes (public) |
| Recruitment | Prospect notes, evaluations, transfer portal | Yes |
| Historical Rosters | Past players, development, outcomes | Yes |
| Schedule/Calendar | Matches, travel, recovery windows | Yes |
| Medical/Injury | Injury history, return-to-play | Yes |

## What We Build

### Core System: RAG (Retrieval-Augmented Generation)

Not a fine-tuned model. A system that:
1. Ingests and indexes all Marshall data
2. Retrieves relevant information based on queries
3. Uses LLM to synthesize answers with citations

**Why RAG over fine-tuning:**
- Works with data you have now
- Updates as new data comes in
- Transparent (shows sources)
- No expensive training runs

### Technical Components

```
┌─────────────────────────────────────────────────────┐
│                   User Interface                     │
│         (Chat / Dashboard / Mobile App)              │
└─────────────────────┬───────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────┐
│                  Query Engine                        │
│     (Understands question, routes to data)          │
└─────────────────────┬───────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────┐
│              Retrieval Layer                         │
│   (Vector search + structured queries + filters)    │
└─────────────────────┬───────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────┐
│                 Data Layer                           │
│                                                      │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐            │
│  │  Wyscout │ │   GPS    │ │Financial │            │
│  └──────────┘ └──────────┘ └──────────┘            │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐            │
│  │Game Plans│ │ Meetings │ │   NCAA   │            │
│  └──────────┘ └──────────┘ └──────────┘            │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐            │
│  │Recruitment│ │ Medical │ │ Schedule │            │
│  └──────────┘ └──────────┘ └──────────┘            │
└─────────────────────────────────────────────────────┘
```

### Example Queries (What It Can Do)

**Opponent Prep:**
- "What are Team X's corner kick tendencies?"
- "How does Team X play against a high press?"
- "Show me Team X's goals conceded this season"

**Performance:**
- "Which players have highest training load this week?"
- "Who's at elevated injury risk based on load history?"
- "Compare Player A's sprint numbers to last month"

**Compliance:**
- "How many training hours have we logged this week?"
- "When does the contact period end?"
- "Are we compliant on scholarship limits?"

**Recruitment:**
- "List prospects we've scouted at left back"
- "What's the status on Recruit X?"
- "Show me transfer portal players that fit our system"

**Institutional Knowledge:**
- "What did we decide about formation against 4-3-3 teams?"
- "What was the scouting report on Player X before we signed them?"
- "What budget do we have remaining for spring travel?"

## Build Sequence

### Phase 2.1: Data Ingestion (Weeks 1-3)
- Connect to Wyscout API (or export data)
- Structure GPS data exports
- Index game plans and documents
- Transcribe meeting recordings
- Parse NCAA rulebook

**Deliverable:** All data in queryable format

### Phase 2.2: Core RAG System (Weeks 4-6)
- Set up vector database (Pinecone, Weaviate, or local)
- Build retrieval pipeline
- Connect to LLM (Claude API, GPT-4, or local)
- Basic query interface

**Deliverable:** Can ask questions, get sourced answers

### Phase 2.3: Specialized Modules (Weeks 7-10)
- Opponent scouting assistant
- Training load dashboard
- Compliance checker
- Recruitment tracker

**Deliverable:** Purpose-built tools for common workflows

### Phase 2.4: Integration & Refinement (Weeks 11-12)
- User feedback loop
- Improve retrieval accuracy
- Add more data sources
- Mobile/quick access options

**Deliverable:** System staff actually use

---

# Part 1: Tracking & Decision Engine

## What We Have

| Component | Status |
|-----------|--------|
| Player detection | Working |
| Ball detection | Working |
| Manual pitch calibration | Working |
| Automatic pitch calibration | Code exists, needs training |
| Basic tracking | Working (loses players in occlusions) |
| Physical metrics | Working |
| Decision engine code | Exists, needs validated tracking |

## What We Need to Build

### Phase 1.1: Automatic Calibration (Weeks 1-6)
- Label Marshall pitch footage for training
- Train HRNet keypoint detector
- Validate on different camera angles

**Deliverable:** Process video without manual point clicking

### Phase 1.2: Tracking Improvements (Weeks 7-12)
- Improve occlusion handling
- Integrate off-screen extrapolation
- Test on full matches

**Deliverable:** More reliable player tracking

### Phase 1.3: Validation (Weeks 13-16)
- Compare to GPS ground truth
- Validate physical metrics accuracy
- Test decision engine outputs with coaches

**Deliverable:** Trusted data that can feed into Part 2

### Phase 1.4: Integration (Weeks 17-20)
- Connect tracking output to Part 2 data layer
- Match data becomes queryable
- "Show me our pressing intensity vs Team X" becomes possible

**Deliverable:** Tracking feeds the Marshall AI

---

# How They Merge

```
Timeline:
─────────────────────────────────────────────────────────────────

Part 2 (Marshall AI):
[===Data Ingestion===][===Core RAG===][===Modules===][===Refine===]
     Weeks 1-3           Weeks 4-6     Weeks 7-10    Weeks 11-12
                              │
                              ▼
                    Immediate Value Starts

Part 1 (Tracking):
[=====Calibration=====][====Tracking====][==Validation==][Integration]
      Weeks 1-6           Weeks 7-12       Weeks 13-16    Weeks 17-20
                                                              │
                                                              ▼
                                              Tracking feeds into Part 2

─────────────────────────────────────────────────────────────────

Week 6:  Part 2 answering questions about Wyscout, GPS, game plans
Week 12: Part 2 fully operational with all existing data
Week 16: Part 1 tracking validated
Week 20: Tracking data flows into Part 2, richer analysis possible
```

---

# Resource Requirements

## Part 2 (Marshall AI)
- Developer time: 1 person, 10-12 weeks
- LLM API costs: ~$100-500/month depending on usage
- Vector database: Free tier or ~$50-100/month
- Infrastructure: Minimal (can run on basic cloud)

## Part 1 (Tracking)
- Developer time: 1 person, 16-20 weeks
- GPU compute for training: ~$500-1000
- Test footage: Marshall practice/match video

## Ongoing
- LLM costs scale with usage
- Maintenance and improvements
- New data source integration

---

# Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Data quality issues | Bad answers | Validate sources, show citations |
| LLM hallucination | Wrong information | RAG grounds answers in real data |
| Tracking doesn't improve | Part 1 delayed | Part 2 works without it |
| Staff don't adopt | Wasted effort | Start with highest-value queries |
| Data access problems | Can't ingest | Work with Marshall IT early |

---

# Success Metrics

## Part 2
- Staff using system weekly
- Questions answered accurately (spot check)
- Time saved on opponent prep
- Compliance issues caught

## Part 1
- Calibration works on new footage without manual work
- Tracking accuracy vs GPS baseline
- Coach validates decision engine outputs make sense

---

# Next Steps

1. **Confirm data access** - What can we actually get from Marshall?
2. **Prioritize queries** - What questions matter most to staff?
3. **Set up data pipeline** - Start ingesting what's available
4. **Build MVP** - Simplest useful version of Part 2
5. **Parallel Part 1** - Begin calibration training with footage

---

# Converting This to Pitch

When we turn this into the Marshall pitch:
- Lead with Part 2 (immediate value)
- Part 1 is "background development that makes it better over time"
- Focus on what they can DO, not how it works
- Show example queries relevant to their workflows
- De-emphasize technical details
- Emphasize: "use data you already have"
