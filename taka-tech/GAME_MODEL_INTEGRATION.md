# Game Model Integration

*How the decision engine learns from and measures Marshall's specific game model.*

---

The engine learns what works. The game model tells it what to look for.

**How it works:**

We feed the engine tracking data from Marshall games. We label moments according to our game model — "this is a BGZ buildup," "this is a counter-press from High Loss," "this is Exploit Space AROUND."

For each moment, the engine:
1. Encodes the situation (defensive shape, player positions, pressure)
2. Finds similar situations from our database
3. Analyzes what actions succeeded vs. failed
4. Builds knowledge: "In BGZ buildups with this defensive shape, switching to the weak side works 65% of the time"

**What the engine learns:**

| Pattern Type | Example |
|--------------|---------|
| Success rates | "Through balls against high lines work 45% in our data" |
| Defensive reactions | "When we switch play, defenders take 2.1s to shift" |
| Player tendencies | "#8 chooses the optimal option 70% of the time in transition" |
| Opponent weaknesses | "Their RCB is slow to recover — target that channel" |

**Analyzing specific game moments:**

Wyscout gives us aggregate stats. Tracking + the engine lets us analyze specific moments:

Example: BGZ buildup at 34:22.
- Engine finds 50 similar situations from our database
- In those situations, switching to weak side led to progression 60% of time
- We played central — which works only 35% in similar situations
- Recommendation: "#6 had the switch available — expected value was 0.15 higher"

This is player-level, moment-specific, grounded in what actually works — not what physics says should work.

---

## Opponent Analysis

This is where the system becomes a genuine competitive advantage.

The same similarity database we build from our own games can be filtered to any opponent. Track their matches, and we learn exactly what works against them — not general tendencies, but specific, actionable patterns:

| What We Learn | Example |
|---------------|---------|
| **Defensive weaknesses** | "Against quick switches, their LB takes 2.8s to recover vs. league average 2.1s — target that channel" |
| **High-value actions** | "Through balls into their left channel succeed 55% vs. 35% on the right — their LCB is slow to drop" |
| **Pressing triggers** | "They press when ball goes back to GK but leave massive gaps when ball goes wide" |
| **Shape vulnerabilities** | "In their 4-4-2, the gap between CM and RB opens when ball is on opposite wing" |
| **Recovery patterns** | "After losing possession, they take 4.2s to reform vs. average 3.1s — we have a window" |

**How it works for game prep:**

Before a match, we query the database: "Show me situations similar to how we want to play, but filtered to this opponent's defensive responses."

The engine returns:
- Situations where teams attacked their weaknesses successfully
- What actions led to high-value outcomes against them specifically
- Where their defensive shape breaks down most often
- Which of our typical patterns would be most effective against their tendencies

**Concrete example:**

*Preparing for Kentucky:*

Query: "Ball in BGZ, opponent in mid-block — what works against Kentucky?"

Results from 8 Kentucky matches analyzed:
- Switch to weak side: 65% progression rate (vs. 45% against average opponent)
- Direct ball into channel: 52% success when their #4 is the nearest defender
- Playing through central midfield: only 28% — their #6 intercepts well
- Recommendation: Force play wide, exploit slow LB recovery, avoid central combinations

This isn't generic scouting ("they defend deep"). It's specific: "In this exact situation, this specific action works X% of the time against them."

**The database grows with every opponent:**

Track 3 matches from an opponent and we have preliminary patterns. Track a full season and we know their tendencies better than they know themselves. Track multiple seasons and we see how they've evolved — and where old weaknesses remain.

---

## The Learning Compounds

Every match we analyze adds to the database. The engine gets smarter over time:
- More situations = better pattern matching
- More outcomes = more accurate success predictions
- Eventually = simulate and search for optimal sequences

This is how it becomes Marshall-specific: it learns from our games, our players, our opponents. But the underlying engine works for any tactical philosophy.

---

## Player Profiles

Over time, the engine builds comprehensive profiles for each player — our guys, recruits, and opponents. Not just tendencies, but measurable attributes:

*Physical:* Top speed, acceleration, turning speed, recovery run intensity, distance covered, sprint frequency, work rate with and without the ball.

*Technical:* Pass speed and accuracy, dribble success rate, first touch quality under pressure, ball retention in tight spaces.

*Decision-making:* How often they choose the highest-xG option, average xG added per action, defenders eliminated per pass, tendency to switch vs hold, pressing trigger timing.

As the system improves, these profiles get richer. We can compare a recruit's profile directly against our current starter: "He's faster but his pass accuracy under pressure is 12% lower. His xG-added per action is similar, but he creates it through dribbling not passing."

Then we can match profiles to opponents: "Our #10 creates +0.15 xG when switching against compact defenses. Their left back has the slowest recovery speed in their back line and gets eliminated 3x more when the ball switches." This is how we find specific mismatches to exploit — not general game plans, but targeted advantages based on actual numbers.

---

*For technical details on the full system architecture, see: TECHNICAL_BRIEF.md*
