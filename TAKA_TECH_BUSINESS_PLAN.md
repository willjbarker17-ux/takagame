# Taka Tech — Business Plan

*Building the decision engine for football.*

---

## Executive Summary

Football analytics tells coaches what happened. Taka Tech tells them what should have happened — and what to do next.

We're building a decision engine that evaluates every option available at every moment of a match, learns what actually works from real data, and generates tactical plans to exploit specific opponents.

**How it works:** We combine tracking data (where all 22 players are) with event data (what actions were taken) to build a database of game situations and outcomes. When a team faces a new situation, the engine finds historically similar situations, analyzes what movements and actions succeeded, and recommends the highest-value option. Over time, this learned knowledge powers a simulation engine that can search multiple moves ahead — finding tactical sequences humans haven't imagined.

**Market:** Professional football clubs. $283M TAM. B2B SaaS at $25K–$150K/year.

**Ask:** $500K seed to validate the system, land professional pilots, and reach $500K+ ARR.

---

## 1. The Problem

The football analytics industry is stuck at "what happened."

Event data (Wyscout, Opta, StatsBomb) tells you a pass was completed. Tracking data (Second Spectrum, SkillCorner) tells you where players were. Neither answers the question that actually matters:

**"Given all 22 players on the pitch at this moment, what was the best option — and how does what we did compare?"**

A midfielder plays a sideways pass. Events say: successful. Tracking shows the positions. But there was a through ball available to the forward worth 0.31 xG. The midfielder chose a pass worth 0.04 xG. That gap — the decision gap — is unmeasured across the sport.

Clubs spend millions on data and analysts, then manually watch film to answer the questions that matter most.

---

## 2. What a Football Decision Engine Does

Given any moment in a match — 22 players, a ball, a score, a context — the engine:

1. **Evaluates the position.** How advantageous is this for the attacking team? Measured by how many defenders are eliminated, proximity to goal, shooting angle, available space, and passing options.

2. **Identifies every option.** All pass targets with interception probability. Through ball opportunities. Dribble lanes. Shot angles. Each option ranked by expected value.

3. **Recommends the best action.** Not just "pass right" — but the specific option with the highest probability of progressing the attack or creating a chance, based on what has worked in thousands of similar situations.

4. **Searches ahead.** If we play this pass, the defense shifts. That opens the weak side. A switch creates a 2v1. The engine explores multi-step sequences, finding paths to goal that aren't obvious from the current moment alone.

5. **Models specific opponents.** Feed it an opponent's recent matches. It learns exactly how they defend — their pressing triggers, their recovery speed, their shape vulnerabilities — and generates plans to exploit those specific weaknesses.

This is the difference between "data" and "intelligence." Event tools describe. Tracking tools measure. The decision engine thinks.

---

## 3. How It Learns: Pattern-Based Learning from Real Data

The engine doesn't rely on designed rules or theoretical models. It learns what works by studying what has actually worked — across thousands of real match situations.

### The Core Idea

Every moment in a football match is a situation: where the ball is, where every player is, how the defense is shaped, how much space exists, what the score is, how deep the defensive line sits, how compact they are, how aggressively they press.

We encode each situation as a set of continuous measurements — not categories like "left wing" or "4-4-2," but precise values: defensive line height in meters, compactness as distance between lines, press intensity as closing speed, ball position as exact coordinates.

When a team faces a new situation, the engine:

1. **Encodes the current state** — ball position, defensive shape, pressure level, attacking positioning, score, time
2. **Searches the database** for historically similar situations — matches where the ball was in a similar area, the defense was shaped similarly, the pressure was comparable
3. **Analyzes what happened next** in those similar situations — what actions were taken, what movements the ball and all 22 players made, and which outcomes were successful
4. **Recommends the highest-value action** based on what actually worked, with a confidence level based on how many similar situations exist and how consistent the outcomes were

### Why Similarity, Not Categories

Traditional approaches put situations into buckets: "zone 14," "high press," "4-4-2 low block." But these categories are imprecise. A ball at 34 meters and 36 meters might fall in different zones despite being tactically identical. "4-4-2" describes hundreds of different defensive shapes.

Similarity matching on continuous features avoids this. The engine measures actual behavior — line height, compactness, pressing speed, spacing — and finds situations that are genuinely similar, without arbitrary boundaries.

### What Makes This Powerful

The engine doesn't need to be told what patterns matter. It discovers them from the data.

If teams that face a compact low block consistently succeed by switching the ball quickly to the weak side, the engine will learn that — not because we coded "switch against low blocks," but because the data shows it. If through balls work against high defensive lines with slow center-backs, the engine will learn that too.

This means the engine can find patterns that coaches haven't articulated, and validate (or challenge) patterns coaches assume are true.

---

## 4. How Useful This Is

### Pre-Match: Opponent-Specific Game Plans

Feed the engine an opponent's last 8–10 matches. It learns how they defend and generates a plan:

- "Against quick switches, their LB takes 2.8s to recover vs. league average 2.1s — target that channel"
- "Through balls into their left channel succeed 55% vs. 35% on the right"
- "After losing possession, they take 4.2s to reform — exploit the transition window"

Not general advice — specific recommendations with specific expected values for this specific opponent.

### Post-Match: Decision Quality Analysis

For every moment where a chance was created, missed, or conceded:

- What options existed
- What each option was worth
- What was actually chosen, and the value difference
- Player-by-player decision quality scores over the match and the season

"We created 2.3 xG. Optimal play would have produced 3.1 xG. Here are the 8 moments where higher-value options were available."

### Player Development

Objective decision-making measurement over time:

- "Player A chooses the highest-value option 72% of the time — up from 61% in August"
- "His through ball recognition is elite. His switch-of-play timing costs 0.08 xG per match"

### Recruitment

Run a prospect's film through the engine:

- Decision quality: how often they choose the best available option
- Physical metrics extracted from video: sprint speed, acceleration, work rate
- Squad fit: how their profile complements or duplicates existing players

### Game Model Accountability

Define your tactical philosophy. The engine measures execution:

- "Are we playing the way we train?"
- "In buildup situations, we're supposed to switch — we do it 45% of the time"
- Subjective "are we playing our way?" becomes a quantified, trackable metric

---

## 5. How It Builds Into the Decision Engine

The pattern-based learning system isn't just a standalone tool. Every match it analyzes produces knowledge that feeds the full decision engine.

### The Path

**Stage 1: Learn what works.** Build the situation database. Encode thousands of possessions. Run similarity matching. Answer: "In situations like this, what actions succeeded?"

**Stage 2: Learn how defenses react.** Track what defenders actually do after each type of action — how the shape shifts, who presses, what spaces open. This replaces guesswork with measured defensive responses.

**Stage 3: Adjust for player ability.** Separate "this worked because it's a good decision" from "this worked because a world-class player executed it." Skill-adjusted success rates make recommendations realistic for any team.

**Stage 4: Simulate ahead.** With learned success rates, learned defensive responses, and realistic player constraints, the engine can now simulate forward: "If we play this pass, the defense will shift like this (we've seen it 200 times), which opens this space, which makes this next action succeed 65% of the time." Multi-step search — exploring action sequences 3–5 moves deep — finds tactical paths that aren't visible from any single moment.

**Stage 5: Self-improvement.** The engine makes predictions. We track what actually happens. It updates. Over thousands of matches, it gets better than any manual analysis. Eventually, it discovers novel tactical sequences through self-play — combinations that emerge from search, not from coaching convention.

Each stage compounds. The engine with 10,000 matches in its database dramatically outperforms the one with 100.

---

## 6. Market

### Primary: Professional Clubs

| Segment | Clubs | Pricing |
|---------|-------|---------|
| Big 5 European leagues | ~100 | $100K–$250K/year |
| MLS, Liga MX, top Americas | ~80 | $75K–$150K/year |
| Championship, 2nd-tier European | ~200 | $50K–$100K/year |
| 3rd-tier, USL, other professional | ~500 | $25K–$75K/year |

Why professional clubs: analytics budgets exist, margins are thin (one extra win justifies six-figure spend), analyst staff can evaluate sophisticated tools, and player salaries dwarf analytics costs ($75K/year for better decisions on a $50M payroll is trivial).

### Secondary: College and Academies

College programs provide validation and case studies. Marshall University (D1, Sun Belt) is the active pilot. Academy adoption follows naturally from parent club adoption.

**Total addressable market: ~$283M.**

---

## 7. Competitive Landscape

| Company | What They Sell | The Gap |
|---------|---------------|---------|
| Hudl | Video platform | No spatial analysis, no decision evaluation |
| Wyscout / Opta | Event database | "Pass completed" — not "was it the right pass?" |
| StatsBomb | Advanced metrics | Still event-level; requires analyst interpretation |
| Second Spectrum / SkillCorner | Tracking data | Raw positions, not intelligence |

Nobody answers: "What should have happened?" That's the gap Taka Tech fills.

**DeepMind's TacticAI** validated similarity-based tactical retrieval in research. Taka is building the product — covering open play, not just set pieces, with a go-to-market for professional clubs.

### Defensibility

The moat compounds with scale:
- **Outcome learning:** more matches analyzed → better predictions → more value → more adoption → more data
- **Opponent database:** more clubs in ecosystem → richer intelligence on every opponent
- **Situation library:** every match adds to the database — 10,000 matches >> 100
- **Switching costs:** each club's accumulated seasons of data make the engine more valuable to them over time

---

## 8. Business Model

| Tier | Price/Year | Includes |
|------|-----------|----------|
| Starter ($25K) | Post-match decision analysis, team-level insights |
| Professional ($75K) | + Player-level reports, opponent modeling, pre-match intelligence |
| Elite ($150K) | + Simulation engine, real-time analysis, full opponent database |

80%+ gross margins. < 6 month payback. 120%+ net revenue retention from tier upgrades.

---

## 9. Go-to-Market

**Months 1–6:** Prove value at Marshall. Build case study and testimonials. Weekly coaching touchpoints.

**Months 6–12:** 2–3 professional pilots (MLS, USL, English lower leagues). Pilot pricing for feedback and testimonial rights. Parallel expansion to 5–10 college programs.

**Year 2–3:** Scale to Championship, Bundesliga 2, Serie B, Ligue 2, MLS. League packages with network effects.

**Year 3+:** Big 5 leagues. Simulation engine operational. Elite tier justified by capabilities no competitor offers.

---

## 10. Financial Plan

### Year 1 Costs: $200K

Engineering ($120K), cloud compute ($25K), data access ($15K), infrastructure ($15K), sales ($20K), legal ($5K).

### Path to Profitability

| Year | Customers | ARR | Costs | Net |
|------|-----------|-----|-------|-----|
| 1 | 7 | $250K | $280K | –$30K |
| 2 | 27 | $1.5M | $650K | +$850K |
| 3 | 65 | $4.5M | $1.8M | +$2.7M |
| 5 | 180 | $15M | $5M | +$10M |

### Seed Round: $500K

Engineering (60%), pilot expansion (20%), sales (15%), operations (5%).

**Milestones:** Tracking validation against GPS. Similarity engine with 1,000+ possessions. 2+ professional pilots. $500K+ contracted ARR.

---

## 11. Why Now

1. **Tracking technology matured** — broadcast-quality player tracking is now feasible outside elite clubs
2. **Event data is commoditized** — the next edge must come from a different layer
3. **Decision analysis is the acknowledged frontier** — coaches know this is what's needed; nobody is selling it
4. **TacticAI proved the concept** — DeepMind validated similarity-based tactical retrieval; Taka is building the product
5. **AI infrastructure is accessible** — GPU compute, model training, and ML tools are available to startups at manageable cost

---

## 12. Relationship to Taka Game

Taka Tech and the Taka board game are separate products under the same parent. They share a founding insight — football is about decisions — but serve different customers. The game targets consumers (DTC + digital). Taka Tech targets professional clubs (B2B SaaS). Each stands on its own.

---

## Summary

**Problem:** Football analytics answers "what happened" but not "what should have happened."

**Solution:** A decision engine that learns what works from real match data, recommends the best action in any situation, and searches ahead to find tactical sequences humans haven't imagined.

**How:** Pattern-based learning on tracking + event data. Encode situations. Find similar historical moments. Analyze what worked. Build into a simulation engine that thinks multiple moves ahead.

**Market:** Professional clubs. $283M TAM. $25K–$150K/year SaaS.

**Ask:** $500K seed. Validate, pilot, reach $500K+ ARR.
