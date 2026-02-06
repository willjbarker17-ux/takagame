# Taka — Business Plan

## Company

Taka is a football decision company. Two product lines, one thesis: football is fundamentally about decision-making under spatial constraints, and that can be productized for both teams and consumers.

- **Taka Tech** (B2B): Intelligence Hub, tracking data, decision engine, and simulation for soccer teams.
- **Taka Game** (B2C): A turn-based football strategy board game (physical + online) that isolates real tactical decisions into a competitive format.

They are separate products with separate platforms. They share a common technical trajectory: both are building toward a simulation-grade understanding of football decisions. Over time, the simulation engine (Taka Sim) becomes the shared core — teams get the full engine, consumers get a constrained version through premium game modes.

---

## The Problem

### For teams (B2B)

Decision quality in football is evaluated informally. Coaches watch film, rely on intuition, and work with limited event data. The tools they have tell them *what happened* — passes completed, shots taken, possession percentage — but not *why* it happened or *what should have happened*.

Existing platforms focus on event tagging and aggregates. They do not capture continuous spatial reality — shape, spacing, timing, and what happens between events. Program knowledge is fragmented across tools (video, stats, GPS, meetings, recruiting notes), making alignment slow and non-cumulative.

No one answers the coaching question: "Was that the right decision?"

### For consumers (B2C)

Football understanding is hard to practice systematically. Most products are arcade simulations or management games. None isolate the actual decision structure of football — the spatial reasoning, passing angles, pressing tradeoffs, and movement coordination that determine outcomes on a real pitch.

---

## The Thesis

A simulation is an advanced game. Both are rulesets that define:

1. **State** — the current position of play
2. **Actions** — what you can do
3. **Transitions** — what happens next
4. **Value** — what counts as good

Taka Game uses hand-designed rules on a discrete grid. Taka Sim will use learned transitions from real tracking data on continuous coordinates. Same four primitives, different fidelity.

The company's arc is to ship both products in parallel, then converge when the simulation reaches a fidelity where consumer game modes can be driven by the same state model and transition dynamics as the professional engine.

---

## Products

### Taka Game (B2C)

A pure strategy game where every decision mirrors real football choices: movement tradeoffs, passing angles, defensive positioning, space creation.

**Core mechanics:**
- 14x10 grid, 22 pieces, two balls (both teams attack simultaneously)
- Movement asymmetry: 3 squares forward without ball, 1 with ball — forces passing to progress
- Facing direction determines 90-degree passing cone and tackle protection
- Chip passes, consecutive passing (combination play), offside, goalie activation
- Every mechanic maps to a real football concept

**Platform:**
- Physical board game edition
- Online edition with matchmaking, ELO rating system, ranked ladder, 16-step interactive tutorial
- Frontend: Next.js (TypeScript + React), backend: Express.js with PostgreSQL

**Monetization (chess.com model):**
- **Free tier:** Play Classic mode with ads, standard ranked ladder
- **Pro tier:** Ad-free, deeper analysis overlays, scenario library, training paths, access to advanced modes as they release

**Roadmap inside the platform:**
- **Classic Mode** (current): discrete grid, deterministic rules. Stays live permanently as the core competitive mode.
- **Bridge Modes** (future): incremental fidelity increases — ball travel time, probabilistic outcomes, player speed profiles — shipped as additional queues/modes inside the same platform.
- **Taka Sim Mode** (convergence): scenario/decision-point experiences powered by the simulation engine. Consumers access a constrained version (limited compute, curated scenario packs, bounded analysis depth).

These are not separate apps. They are modes inside one platform — like different time controls in chess.

### Taka Tech (B2B)

A three-layer product suite for soccer teams.

**Layer 1: Intelligence Hub**

A program-specific knowledge system that connects scattered data into one queryable source of truth.

What it connects:
- Wyscout (match events, opponent tendencies, player stats)
- GPS/Catapult (training load, distances, sprints, fatigue patterns)
- Game plans and tactical documents
- Meeting recordings (transcribed, searchable)
- Recruitment notes and evaluations
- Financial data, medical/injury history, schedule, compliance

What it does:
- Cross-source pattern recognition ("When we play high-press teams, do our players fatigue more?")
- Institutional memory that compounds across seasons instead of resetting with staff turnover
- Proactive alerts (injury risk patterns, compliance limits, opponent changes)
- Auto-generated prep documents pulling from all sources

This is the operational wedge. Value comes from alignment and compounding institutional knowledge. It works with data teams already have.

**Layer 2: Tracking**

Computer vision extracts player and ball coordinates from match video to generate spatial data teams don't have from event feeds alone.

Current state: 36,000+ lines of code. Player detection, ball detection, basic tracking, physical metrics all working. Main challenges remaining: automatic pitch calibration across camera angles, identity persistence through occlusions and cuts, and validation against GPS ground truth.

What it produces: frame-by-frame XY coordinates in meters, plus derived physical metrics (speed, distance, acceleration, sprints). This feeds into the Hub as a new class of proprietary spatial evidence.

**Layer 3: Decision Engine / Taka Sim**

The core intelligence layer. Starts as similarity-based retrieval, evolves toward simulation-grade scenario analysis.

Phase 1 — Similarity retrieval:
- Encode game situations as continuous features (defensive shape, ball position, pressure level, attacking width, players in box, context)
- Find similar situations from a database of real matches
- Report what actions succeeded and at what rates
- "In situations like this, passes to the wing succeed 65% of the time"

Phase 2 — Outcome learning:
- Learn actual success distributions from tracking data
- Opponent-specific filtering ("what works against Kentucky's defensive shape")
- Player-level evaluation ("did they choose the highest-value option?")

Phase 3 — Taka Sim (scenario search):
- Use learned transition models to simulate forward from decision points
- Search for optimal action sequences through possessions, transitions, set pieces
- Variable depth: prune bad branches, go deep on promising ones
- Not full 90-minute match simulation — decision-point solving

**What teams get (that consumers do not):**
- Full situation database and opponent-filtered queries
- Bulk data exports and API access
- Unlimited compute and deep tree search
- Coaching-grade preparation workflows and integrations
- Proprietary tracking outputs
- Custom game model configuration

---

## Marshall Partnership

Marshall University Men's Soccer is the design partner and first proof case.

### Why Marshall

- Division I program (Sun Belt Conference) with real operational workflows and data
- Provides the ground truth for what outputs matter — real coaching feedback, not vanity metrics
- Supplies match footage for tracking development and validation
- Becomes the first reference case: "built with a competitive D1 program; validated in real prep and review"

### Deal Structure

**Category exclusivity:** Marshall receives exclusivity for NCAA Division I for a defined term. Taka Tech is not sold to other D1 programs during this period. In exchange, Taka focuses B2B sales on professional clubs (and potentially academies, federations).

This achieves:
- Marshall gets a protected competitive edge in their category
- Taka retains access to the larger, more financially viable pro club market
- Both sides have aligned incentives — Marshall's success is Taka's proof case

(Exact scope and term to be negotiated. The structure is: exclusivity in college soccer, freedom in professional soccer.)

### Marshall Pilot Timeline

**Phase 1 — Intelligence Hub (weeks 1-6):**
- Weeks 1-2: Data ingestion — connect docs, meeting transcripts, Wyscout exports, scouting reports
- Weeks 3-4: Core system working — staff can start querying
- Weeks 5-6: Refinement based on staff feedback, improve retrieval quality
- Success criteria: staff uses it weekly without prompting, saves time, surfaces real cross-source connections

**Phase 2 — Tracking (months 1-3):**
- Month 1: Automatic pitch calibration on Marshall footage
- Month 2: Tracking refinement for real match conditions
- Month 3: Validation against GPS ground truth
- Success criteria: repeatable on Marshall footage, accuracy validated, outputs feed into Hub

**Phase 3 — Decision Engine (months 4-6):**
- First scenario analysis outputs used in prep and review
- Retrieval-based: "in similar situations, X worked Y%"
- Game model integration: engine measures execution of Marshall's tactical principles
- Success criteria: engine flags same moments coaches flag (face validity), stable metrics across matches, actionable opponent-specific insights

---

## How the Products Connect

Taka Game and Taka Tech are not "a game studio plus an analytics startup." They are two product lines building the same thing from different directions. To understand why, you have to understand what the simulation engine actually is.

### The simulation engine is a game tree search

The end goal of Taka Tech is a simulation engine that can take any football situation and find the optimal path to a scoring opportunity. Here is what that looks like computationally:

```
Ball reception moment
    ├── Pass to Player A
    │      ├── A shoots → evaluate xG         ← terminal, score it
    │      ├── A passes to B → keep exploring
    │      └── A loses ball → stop            ← terminal, score it
    ├── Pass to Player B (promising) → go DEEP
    ├── Dribble into pressure → PRUNE early
    └── Shot → evaluate xG                    ← terminal, score it
```

This is tree search. Explore branches. Prune bad ones. Go deep on promising ones. Evaluate terminal states. Return the best path.

This is also exactly what a game AI does. Chess engines, Go engines, and Taka Game AI all solve the same computational problem: given a state, search through possible action sequences to find the one with the highest expected value.

The simulation engine IS a game engine. The difference is where the rules come from:

| | Taka Game | Taka Sim |
|---|-----------|----------|
| **State** | Discrete grid + facing + possession | Continuous coordinates + pressure/shape features |
| **Actions** | Move / turn / pass / tackle / shoot | Same categories, richer parameters |
| **Transitions** | Hand-designed deterministic rules | Learned from real tracking data |
| **Value** | Scoring / reaching advantageous positions | xT, xG, progression, defensive stability |
| **Search** | Tree search over game rules | Tree search over learned transition models |

Same structure. Same algorithms. Different transition model underneath.

### The core integration: Taka Game is the development environment for the search engine

The simulation engine has two halves:

1. **The learned model** (Phases 1-4): what actually happens in real football — pass success rates, defensive reactions, movement physics. This comes entirely from tracking data. The game does not contribute here.

2. **The search engine** (Phase 5): given the learned model, explore action sequences to find optimal paths. This is tree search, position evaluation, branch pruning, multi-agent reasoning.

The game directly develops half #2.

Building AI for Taka Game requires solving the same problems the simulation search engine needs:

**Tree search strategies.** How deep to search. When to prune a branch. How to allocate compute to promising lines vs exploring alternatives. Variable-depth search that goes deep on high-value branches and cuts bad ones early. These algorithms are identical whether the underlying state is a 14x10 grid or continuous pitch coordinates — you develop them on the game, then port them to the simulation.

**Position evaluation.** "How good is this state?" The game needs a function that scores any board position — are you in a strong attacking position? Is your defense exposed? Is this a winning state or a losing one? The simulation needs the exact same capability, just with different input features. The architecture of the evaluation system — how to combine spatial features into a single value score — transfers directly.

**Action ranking and selection.** From any state, which actions are worth exploring? You can't search every possible branch — you need heuristics that identify the most promising options first. The game AI trains these ranking systems at scale: millions of positions, clear outcomes, fast iteration. The ranked action structures carry over to the simulation.

**Multi-agent reasoning.** Both the game and the simulation involve two sides making decisions. What will the opponent do? How does that affect the value of my action? Modeling opponent responses is the same problem in both environments — the game lets you develop and test these models in a controlled setting.

**Monte Carlo methods.** Both use Monte Carlo Tree Search (MCTS) or similar sampling-based approaches to handle the branching complexity. The game is where you tune rollout policies, exploration vs exploitation tradeoffs, and convergence criteria before applying them to the harder simulation problem.

In concrete terms: you build a Taka Game AI that can search 10+ moves ahead, evaluate positions, prune intelligently, and reason about opponent responses. Then when the learned transition model from tracking data is ready, you plug it into the same search infrastructure. The search engine is already built and tested.

This is not a loose analogy. It is the same codebase, the same algorithms, applied to a different transition model.

### Additional technical benefits

Beyond the search engine, Taka Game provides three more concrete advantages:

**1. Controlled evaluation harness**

The simulation engine needs continuous benchmarks. Waiting for real match cycles is too slow. The game supplies a stable testbed where you can measure progress:
- Can the engine rank options in a way strong players agree with?
- Does engine guidance measurably improve outcomes in standardized scenarios?
- Can it detect and score decision quality reliably?
- Do model changes improve or regress these metrics?

Every change to the search algorithms or evaluation functions can be A/B tested in the game environment instantly. This is an internal "unit test suite" for tactical reasoning — faster iteration cycles, measurable progress.

**2. Human decision data that improves search efficiency**

Gameplay generates large-scale labeled data on:
- Which actions humans consider from a given state (branch ordering priors)
- Which lines humans perceive as high-value (search prioritization)
- Common tactical motifs (scenario library design)
- Where humans consistently make suboptimal choices (teaching content)

This data directly improves search efficiency in the simulation: if you know which branches humans typically consider first, you can order the simulation's search to explore those branches first, reaching good solutions faster.

Important: tracking data remains the source of truth for success distributions and movement physics. Game data helps search efficiency and content — it does not replace real-match grounding.

**3. Scenario and explanation UX — solved in a live environment**

Taka Sim's success depends on interface questions:
- How to present a situation quickly
- How to show options and tradeoffs without overwhelming users
- How to explain "why" an action is better (risk, confidence, downstream value)
- How to let users explore alternatives ("what if I had passed here instead?")

The game platform provides high-volume iteration on exactly these problems. This is direct product leverage — it makes the simulation engine usable when it arrives.

### What this means concretely

The development sequence looks like this:

1. **Now:** Build Taka Game AI using tree search, position evaluation, MCTS. The game is simple enough to iterate quickly — deterministic rules, discrete state, fast computation.

2. **In parallel:** Taka Tech learns football reality from tracking data (Phases 1-4). Pass success rates, defensive reactions, movement physics, state transitions — all measured from real matches.

3. **Convergence:** The learned transition model plugs into the search infrastructure already developed and tested through the game. The simulation engine doesn't need to build search from scratch — it inherits a proven search system and just swaps the underlying model.

This is why it's one company. Taka Game is not a side business that happens to share a brand. It is actively building half of the simulation engine's technical stack.

### The convergence point

Convergence is defined precisely: it happens when the simulation engine's learned state model (how players move, how the ball travels, how actions succeed or fail) can drive a playable mode inside the consumer platform — meaning the same search engine that powers team analysis also powers consumer scenario gameplay, just with different access and compute limits.

At that point:
- Taka Sim Mode becomes a consumer-facing scenario experience where ball and player dynamics are calibrated from real tracking data
- The same engine powers both the consumer analysis mode and the team's professional tools
- The difference is access and compute, not the underlying model

Until convergence, the products operate independently and succeed on their own terms. Taka Game does not need Taka Sim to be fun and competitive. Taka Tech does not need Taka Game to deliver value to teams. But every improvement to the game AI is direct progress toward the simulation engine.

### The flywheel

1. **Taka Game develops the search engine and builds brand** — tree search, evaluation, MCTS, plus a football-native audience and credibility in the football world.
2. **Taka Tech develops the learned model** — tracking + similarity matching + outcome learning = the transition model that grounds the simulation in reality.
3. **When both are ready, they combine** — the search engine (from Game) + the learned model (from Tech) = Taka Sim. Teams get the full engine. Consumers get a constrained version through premium modes.
4. **Sim upgrades both products** — better AI opponents and analysis for the game, better decision intelligence for teams. Revenue and credibility compound on both sides.

---

## Market

### B2B (Taka Tech)

**Professional clubs (primary):**
- ~500 professional clubs in top European leagues
- ~150 MLS, Liga MX, other Americas leagues
- ~200 top clubs in other regions
- ~3,000 second/third tier professional clubs

**Academies and federations (secondary):**
- 200+ professional club academies
- National federations
- Elite youth clubs

(College soccer excluded during Marshall exclusivity period.)

### B2C (Taka Game)

**Target audience:**
- Football fans who want to engage with the tactical side of the sport
- Strategy game players (chess, Go) looking for a new competitive format
- Coaches, players, and analysts who want to sharpen decision-making

**Market analogy:** Chess.com has 100M+ registered users and generates significant revenue through a free + premium model on a pure strategy game. Taka targets a fraction of this in the football-specific niche — but the football audience is massive.

---

## Business Model

### Taka Game (B2C)

| Tier | Price | Includes |
|------|-------|----------|
| **Free** | $0 (ads) | Classic mode, standard ranked ladder, basic puzzles |
| **Pro** | TBD/month | Ad-free, analysis tools, scenario library, training paths, advanced modes |

### Taka Tech (B2B)

| Product | Pricing Model | Includes |
|---------|---------------|----------|
| **Intelligence Hub** | Subscription (seat or program-based) | Knowledge system, cross-source retrieval, alerts, prep docs |
| **Tracking** | Tiered by matches processed | Spatial data extraction, physical metrics, Hub integration |
| **Decision Engine** | Add-on module | Scenario analysis, opponent prep, game model measurement |
| **Full Suite** | Enterprise | All products + custom integrations + dedicated support |

Revenue does not depend on simulation maturity. The Hub delivers value independently. Tracking delivers value independently. The decision engine layers on top.

---

## Competitive Landscape

### B2B

| Company | What They Do | Limitation |
|---------|-------------|------------|
| **Hudl** | Video platform, basic tagging | No decision analysis, no knowledge integration |
| **Wyscout** | Scouting database, event data | Events only, no spatial analysis |
| **StatsBomb** | Advanced event metrics | Still event-level, no real-time tracking |
| **Second Spectrum** | Tracking data | Raw data provider, not insight layer |
| **SkillCorner** | Broadcast tracking | Data provider, requires separate interpretation |

**Taka Tech differentiation:**
1. Decision-level analysis — not "pass completed" but "was it the right pass"
2. Knowledge integration — connects all program data, not just match stats
3. Game model adaptation — evaluates execution against the team's own philosophy
4. Outcome learning — improves with each team's data, becomes team-specific

### B2C

| Product | Category | Limitation |
|---------|----------|------------|
| **FIFA/EA FC** | Arcade simulation | Reflexes over decisions, no tactical depth |
| **Football Manager** | Management sim | Administrative, not spatial/tactical |
| **Hattrick/OSM** | Browser management | Abstract, no on-pitch decision-making |

**Taka Game differentiation:**
1. Pure strategy — every decision mirrors real football choices
2. Tactical authenticity — mechanics map directly to real concepts (body orientation, pressing angles, offside, combination play)
3. Competitive depth — high skill ceiling, ELO rating, ranked ladder
4. Accessible — rules learned in 10 minutes, depth emerges from interaction of simple rules

---

## Go-to-Market

### B2C (Taka Game) — start with Classic, grow the community

1. Ship the online Classic mode with matchmaking, ELO, and tutorial
2. Build a competitive community around football decision-making
3. Position as a strategy game with football authenticity — not an arcade sim
4. Monetize through free tier (ads) + Pro tier (ad-free + analysis)
5. Release bridge modes over time as additional queues

The role of B2C is: brand, revenue, and a living lab for decision UX.

### B2B (Taka Tech) — start with Hub (fast value), build toward engine (moat)

1. Deploy Intelligence Hub to Marshall — prove adoption and value in weeks
2. Use Marshall as the design partner to validate workflows, retrieval quality, and adoption
3. Layer in tracking as the technical capability matures and is validated
4. Expand to professional clubs once the Marshall proof case is established
5. Add decision engine outputs once tracking is reliable and coaches trust the data

For credibility, anchor the data stack narrative with known categories teams already use (Wyscout, StatsBomb, SkillCorner), while positioning Taka as the system that connects knowledge + spatial reality + decision analysis.

---

## Roadmap

### Parallel from day 1

| Track | Near-term (months 1-6) | Mid-term (months 6-12) | Longer-term |
|-------|------------------------|------------------------|-------------|
| **Taka Game** | Online Classic live, matchmaking, ELO, tutorial, community growth | Pro tier launch, first bridge mode, retention optimization | Additional modes, Taka Sim Mode (at convergence) |
| **Taka Tech** | Marshall Hub deployed + adopted, tracking calibration + validation | Decision engine v1 producing scenario outputs, first pro club conversations | Engine expansion, opponent modeling, simulation search, additional team deployments |

### Convergence milestone

Taka Sim becomes a stable scenario-analysis mode using the same simulation state dynamics as the team engine. Consumer access is constrained to Pro-tier capabilities — bounded compute, curated scenarios, limited analysis depth.

This milestone is gated on: tracking reliability validated, transition models learned and stable, scenario UX refined through game platform iteration, engine recommendations matching coach judgment (face validity).

---

## Key Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| "Two products" perception | Shared decision primitives, explicit convergence milestone, consistent language, one brand |
| Tracking reliability | Validation-gated — don't build on top of unreliable data; Hub delivers value independently |
| Simulation overclaim | Scenario-first wedge (not 90-minute matches); empirical retrieval + confidence scoring; honest about what's learned vs what's assumed |
| Team adoption inertia | Hub delivers value in weeks, not months; build trust with operational alignment before layering advanced capabilities |
| Game doesn't gain traction | Game can succeed or fail independently of Tech; Tech doesn't depend on Game for revenue or proof |
| Marshall relationship | Clear exclusivity terms protect both sides; pilot success criteria defined upfront |

---

## What We're Asking Investors to Believe

1. Football decision-making can be productized as a scalable platform — for both teams and consumers.
2. A retrieval-first approach grounded in real tracking data is a credible path to simulation-grade decision analysis.
3. A consumer strategy game can materially accelerate productization, evaluation, and distribution — without pretending consumer data replaces real-match grounding.
4. Marshall as a design partner provides the validation loop needed to build a product teams will trust.

---

## Current Traction

- **36,000+ lines** of production code across tracking pipeline and decision engine
- **Working board game** with full rule implementation, interactive tutorial, online play
- **Active pilot relationship** with Marshall University Men's Soccer (Division I, Sun Belt Conference)
- **Technical foundation:** Player/ball detection, tracking pipeline, pitch calibration, physical metrics, interactive tactical analysis, decision engine modules

---

## Long-Term Vision

Year 1: Prove the Hub delivers operational value to a real program. Prove Classic Taka retains players and builds a competitive community.

Year 3: Taka Tech is the decision layer for serious programs — "We use tactical intelligence the way we use GPS tracking." Taka Game has a loyal competitive community and generates meaningful B2C revenue.

Year 5: Taka Sim exists. Teams use it for opponent preparation and player evaluation. Consumers experience it as the most tactically authentic football game ever made. The engine that powers both is the same — one company, two surfaces, compounding.

The way football understands decisions changes. Not "did the pass work" but "was it the right pass given all options." That question gets answered the same way whether you're a coach at a professional club or a player in a ranked Taka match.
