# An AI for Football — Business Concept

## The One-Liner

A ChatGPT built on top of everything a football club already knows — turning scattered data, documents, and conversations into one queryable brain.

---

## The Problem

Football clubs produce an enormous amount of information every single week. The issue is not a lack of data. The issue is that all of it lives in different systems that do not talk to each other, and the people who need to make decisions cannot access it fast enough or connect it meaningfully.

A typical professional or high-level college program has data spread across a dozen or more platforms. The assistant coach preparing for Saturday's opponent has to manually check Wyscout for event data, pull up GPS reports in a separate system, dig through old scouting documents in a shared drive, and try to remember what the head coach said in a meeting three weeks ago about how to deal with a low block. None of these systems are linked. The coach becomes the integration layer — and the coach's memory is the search engine.

This is the same problem that has created billion-dollar businesses in other industries. Legal teams had documents everywhere — now they have AI that searches across all of them. Sales teams had CRM data, emails, and call recordings scattered — now they have AI that connects it. Healthcare, finance, consulting — every knowledge-heavy industry is building AI systems that sit on top of existing data and make it queryable.

Football has the same problem. It just hasn't had the same solution yet.

---

## What Data Do Football Clubs Actually Have?

This is the foundation of the business. You are not asking clubs to generate new data. You are connecting what they already produce.

### Match & Performance Data

| Source | What It Contains | Who Has It |
|--------|-----------------|------------|
| **Wyscout / StatsBomb / Opta** | Match events — every pass, shot, tackle, duel, set piece. Player-level and team-level statistics. Opponent data for scouting. | Nearly every professional club and most top-tier college programs subscribe to at least one |
| **GPS & Physical Tracking** (Catapult, STATSports, Polar) | 200-300+ metrics per player per session — total distance, high-speed running, sprint counts, accelerations, decelerations, metabolic load, fatigue indices | Standard at professional level, widespread at D1 college level |
| **Optical Tracking** (Second Spectrum, SkillCorner, Hawkeye) | Player coordinates at 25 fps across the full pitch — positional data, off-ball movement, spacing, shape | Available at top leagues, increasingly accessible via broadcast-derived tracking |
| **Video** (Wyscout, Hudl, SBG, club cameras) | Full match footage, training footage, clip collections tagged by coaches | Universal — every club has match video |

### Staff Knowledge & Documents

| Source | What It Contains | Current State |
|--------|-----------------|---------------|
| **Game plans & tactical documents** | Formation notes, pressing triggers, buildup structures, set piece designs, opponent analysis | Typically in Google Docs, PowerPoint, PDFs, or bespoke analysis software — rarely indexed or searchable after the week ends |
| **Scouting & recruitment reports** | Prospect evaluations, opposition scouting notes, agent communications, shortlists | Scattered across email, spreadsheets, scouting platforms (Wyscout, TransferRoom), personal notebooks |
| **Meeting recordings & discussions** | Pre-match briefings, post-match reviews, halftime talks, recruitment meetings, board discussions | Often recorded but almost never transcribed, indexed, or made searchable — the richest source of context and reasoning, and the most underutilized |
| **Training session plans** | Drill designs, session objectives, periodization plans, individual development programs | Usually in coaching software (e.g., Tactics Manager) or personal files — disconnected from match outcomes |

### Player Management Data

| Source | What It Contains | Current State |
|--------|-----------------|---------------|
| **Medical & injury records** | Injury history, diagnosis, treatment protocols, return-to-play timelines, surgery records | In medical management systems (e.g., Zone7, Kitman Labs) or spreadsheets — siloed from performance data |
| **Wellness & readiness** | Daily questionnaires — sleep quality, muscle soreness, fatigue, mood, RPE (rate of perceived exertion) | Collected via apps (e.g., Smartabase, Catapult) but rarely connected to tactical or match data |
| **Contracts & financials** | Wages, transfer fees, agent fees, contract expiry dates, bonus structures, budget allocations | In finance systems or spreadsheets — used by sporting directors and executives |

### Communication & Context

| Source | What It Contains | Current State |
|--------|-----------------|---------------|
| **Emails & messages** | Staff discussions about players, transfer targets, tactical ideas, scheduling | Completely unstructured, unsearchable, and lost when staff members leave |
| **Player feedback & reviews** | Performance reviews, development conversations, behavioral notes | Often informal, rarely documented systematically |
| **Board & executive discussions** | Strategic direction, budget decisions, philosophy alignment | Captured in meeting minutes at best |

**The key insight:** Clubs are already paying to generate all of this data. They are just not getting the full value from it because no system connects it.

---

## How It Works

The product is conceptually simple. It is a knowledge layer that sits on top of all of a club's existing data and makes it conversational.

### The Architecture (Non-Technical)

```
┌─────────────────────────────────────────────────┐
│                   The Interface                  │
│                                                  │
│   Staff ask questions in plain language.         │
│   The system answers from the club's own data.   │
│                                                  │
└─────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────┐
│                  The AI Layer                    │
│                                                  │
│   Understands the question, searches across      │
│   all connected data sources, synthesizes an     │
│   answer grounded in the club's own information. │
│                                                  │
└─────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────┐
│               Connected Data Sources             │
│                                                  │
│   Wyscout  ·  GPS  ·  Video tags  ·  Game plans │
│   Meetings ·  Medical · Scouting  ·  Contracts   │
│   Training ·  Wellness · Emails   ·  Financials  │
│                                                  │
└─────────────────────────────────────────────────┘
```

### What It Looks Like in Practice

**A coach preparing for the weekend match asks:**

> "What do we know about FC Dallas's pressing structure? How have we played against teams that press like them this season?"

The system searches across:
- Wyscout event data on FC Dallas (pressing intensity, PPDA, pressing triggers)
- The club's own scouting reports on FC Dallas from previous encounters
- Meeting transcripts where staff discussed FC Dallas or similar pressing teams
- GPS data from matches against high-pressing opponents (to flag physical load implications)
- Past game plans the club used against similar pressing structures

It returns a synthesized answer — not a data dump, but a coherent response that connects information the coach would have needed 45 minutes and four different platforms to assemble manually.

**A sporting director evaluating a transfer target asks:**

> "Show me everything we have on this player. How does he compare to our current options at that position?"

The system pulls together:
- Wyscout statistical profile and match footage clips
- Internal scouting reports and evaluations from staff who watched him
- Any mentions from recruitment meetings (transcribed)
- Contract and financial context (fee range, wage expectations, agent)
- Comparison metrics against current squad players at that position from GPS and event data

**A fitness coach reviewing weekly load asks:**

> "Which players are above their rolling 4-week average for high-speed running, and do any of them have a history of hamstring issues?"

The system connects:
- GPS data (high-speed running distance, acute-to-chronic workload ratios)
- Medical records (injury history filtered for hamstring)
- Wellness questionnaire data (recent soreness or fatigue flags)

**A head coach after a loss asks:**

> "We conceded three goals from crosses this month. Is this a trend? What were we doing differently in the games where we defended crosses well?"

The system looks at:
- Wyscout event data on crosses conceded across all recent matches
- Tactical documents for the matches in question (formation, personnel)
- Meeting transcripts where defensive setup was discussed
- GPS data comparing defensive workload and positioning metrics across the sample

---

## Why This Is a Business

### The Market Exists

Every knowledge-heavy industry is adopting this pattern:

| Industry | The AI Layer | Examples |
|----------|-------------|----------|
| Legal | AI that searches across all case files, contracts, precedents | Harvey, CoCounsel |
| Sales | AI that connects CRM data, calls, emails, deal history | Gong, Clari |
| Healthcare | AI that integrates patient records, research, imaging | Abridge, Hippocratic |
| Consulting | AI that searches past projects, frameworks, deliverables | Internal tools at McKinsey, BCG |
| **Football** | **AI that connects match data, scouting, meetings, GPS, medical** | **This does not exist yet at scale** |

Football is a knowledge-heavy, data-rich, decision-intensive industry. The same pattern applies.

### The Business Model

**Primary revenue: SaaS subscription per club.**

Pricing tiers based on:
- Level of play (professional vs. college vs. academy)
- Number of data sources connected
- Number of staff seats
- League / federation licensing

**Potential pricing ranges:**
- Professional clubs (MLS, Championship, Liga MX, etc.): $3,000–$10,000/month
- Top college programs (D1 Power conferences): $1,000–$3,000/month
- Lower divisions / academies: $500–$1,500/month

**Why clubs pay:**
- They are already paying $50K–$200K+/year on data subscriptions (Wyscout, GPS systems, medical platforms) and getting fragmented value from each one individually
- This product makes every existing subscription more valuable by connecting them
- Staff time saved on manual research is immediately quantifiable
- Competitive information advantage is difficult to put a price on but easy to feel

### The Moat

1. **Data network effects.** The more a club uses the system, the more context it accumulates. Switching costs increase over time — the AI has learned the club's terminology, game model, and history. A new product would start from zero.

2. **Football-specific understanding.** A generic ChatGPT does not know what PPDA means, cannot interpret a Wyscout CSV intelligently, and has no context on periodization or pressing triggers. The football-specific layer — how the AI interprets and connects football data — is specialized knowledge that takes time to build.

3. **Integration complexity.** Building reliable connectors to Wyscout, Catapult, StatsBomb, STATSports, Hudl, and every other platform clubs use is unglamorous but defensible work. Each integration is a barrier to entry.

4. **Institutional knowledge as lock-in.** After two seasons, the system contains a club's entire tactical history, scouting archive, and meeting record. That is not data the club wants to lose or migrate.

---

## What Makes Football Different From Other Industries

Football has specific characteristics that shape how this product must work:

### The weekly cycle is king
Football operates in compressed 7-day (or less) cycles. Match → recovery → analysis → preparation → match. The product must deliver value within this rhythm. A coach does not need a 30-page report on Monday — they need an answer to a specific question in 30 seconds.

### Decisions are high-frequency and high-stakes
A sporting director might evaluate 200 players to make 3 signings. A coach makes 50+ in-game tactical decisions per match. A medical team manages load for 25+ players daily. The volume of decisions is relentless, and the quality of information behind each one matters.

### Context matters more than raw numbers
Football analytics has historically focused on producing metrics — xG, xA, PPDA, progressive passes. But coaches do not think in isolated metrics. They think in context: "We struggled to progress through their press in the first 20 minutes because their #6 was cutting passing lanes and our fullbacks weren't offering width early enough." A useful AI system must understand and work with this kind of contextual, situational thinking — not just return numbers.

### Knowledge walks out the door constantly
Player turnover, staff turnover, and the short tenure of coaching positions mean clubs lose institutional knowledge continuously. The average coaching tenure in professional football is roughly 1-2 years. Every time a coach leaves, they take their tactical reasoning, opponent knowledge, and player assessments with them.

### Multi-language and multi-cultural environments
Professional football clubs routinely have staff and players speaking 3-5+ languages. Meeting discussions happen in multiple languages. Scouting reports come in from scouts based in different countries. The system must handle this natively.

---

## The Competitive Landscape

### What exists today

| Category | Examples | Limitation |
|----------|----------|------------|
| **Event data platforms** | Wyscout, StatsBomb, Opta | Data delivery only — no cross-source intelligence, no club-specific context |
| **Video analysis** | Hudl, SBG (now Sportscode), Catapult Video | Tagging and clipping — does not connect to GPS, medical, or tactical documents |
| **GPS & physical** | Catapult, STATSports, Polar | Physical data only — completely siloed from tactical and scouting data |
| **Medical & wellness** | Zone7, Kitman Labs | Injury prediction and load management — does not integrate tactical or scouting context |
| **Recruitment platforms** | Wyscout, TransferRoom, SciSports | Player discovery and comparison — disconnected from internal staff evaluations and financial constraints |
| **General AI tools** | ChatGPT, Gemini, Claude | No football domain knowledge, no access to club data, no integration with football platforms |

**The gap:** Every existing tool serves one vertical. No product connects them into a single queryable system. The club's staff is still the integration layer.

### Why incumbents have not built this

- Wyscout, Catapult, and Hudl are incentivized to keep clubs inside their own ecosystem, not to become a layer that connects competitors' data.
- Data companies sell data. They do not have the AI expertise or the incentive to build a cross-platform knowledge layer.
- General AI companies (OpenAI, Google, Anthropic) build horizontal platforms. They will not build football-specific data integrations, domain understanding, and club-specific learning.

This is a classic gap for a vertical AI product.

---

## Go-To-Market

### Start narrow, expand with usage

**Phase 1: Core knowledge system**
- Connect the 3-4 highest-signal data sources (event data, documents, meeting transcripts)
- Deliver a working, queryable system within weeks
- Land 5-10 design partner clubs who help shape the product

**Phase 2: Expand data sources**
- Add GPS, medical, wellness, financial integrations
- Cross-source intelligence becomes the differentiator
- Clubs start experiencing insights they could not get from any single platform

**Phase 3: Platform effects**
- Clubs have 2+ seasons of accumulated knowledge in the system
- Switching costs are real — the AI understands the club's language, history, and model
- Word of mouth in football is powerful — staff move between clubs and bring the product with them

### The staff movement flywheel

This is unique to football. Coaches and analysts change clubs frequently. When an analyst who used this product at Club A moves to Club B, they ask for it. This is exactly how platforms like Hudl and Wyscout spread — through the movement of people who became dependent on the tool.

---

## Risks and Honest Challenges

| Risk | Reality |
|------|---------|
| **Data access & integrations** | Wyscout, Catapult, etc. have APIs but they are not always open or well-documented. Building reliable integrations is real engineering work. Some platforms may resist. |
| **Trust and adoption** | Football staff are busy and skeptical of technology that promises to "revolutionize" their work. The product must save time from day one, not require a learning curve. |
| **Data sensitivity** | Clubs are protective of their data — tactical plans, scouting targets, medical records, financials. Security, privacy, and data isolation between clubs must be bulletproof. |
| **Football domain complexity** | A generic AI will give generic answers. The system needs genuine football understanding — tactical concepts, periodization logic, scouting evaluation frameworks — to be useful rather than annoying. |
| **Club budgets** | Outside the top leagues, budgets for technology are tight. Pricing must reflect the value delivered at each level. |

---

## Summary

Football clubs are knowledge organizations that operate under extreme time pressure. They generate vast amounts of data across many disconnected systems and rely on human memory to connect it all. This is the same structural problem that has created large AI businesses in legal, healthcare, sales, and consulting.

The product is an AI knowledge layer purpose-built for football. It connects the data clubs already have — event data, GPS, video, scouting, meetings, medical, financial — into one system that staff can query in plain language. It learns the club's own terminology, game model, and history over time, becoming more valuable the longer it is used.

The market is ready. The data exists. The tools to build it exist. No one has assembled the football-specific version yet.
