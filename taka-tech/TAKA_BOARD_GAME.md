# Taka — The Board Game

## Overview

Taka is a turn-based football strategy game that distills real tactical decision-making into an accessible, competitive format. Unlike simulation games (FIFA) or management games (Football Manager), Taka is a pure strategy game where every decision mirrors real football choices: movement trade-offs, passing angles, defensive positioning, and space creation.

**The core insight:** Football is fundamentally about decision-making under spatial constraints. Taka isolates these decisions in a way that's learnable, strategic, and competitive.

---

## The Board

### Dimensions

- **14 rows × 10 columns** (140 total squares)
- Rows numbered 0-13 (bottom to top from white's perspective)
- Columns numbered 0-9 (left to right)

### Goal Areas

- **White's goal:** Row 0, columns 3-6 (4 squares)
- **Black's goal:** Row 13, columns 3-6 (4 squares)
- Only goalies can enter goal areas

### Zones

The pitch is divided into three zones:

| Zone | Rows | Purpose |
|------|------|---------|
| Black's shooting zone | 0-4 | Black can shoot from here |
| Middle zone | 5-8 | Neutral territory |
| White's shooting zone | 9-13 | White can shoot from here |

---

## Pieces

### Setup

Each team has **11 pieces**:
- 10 outfield players
- 1 goalie (starts off the board, activated when needed)

### Piece Properties

Each piece has:

1. **Color** — White or black (team identification)
2. **Position** — Current square on the board
3. **Facing direction** — North, south, east, or west
4. **Ball possession** — Whether the piece has the ball

### Starting Positions

Teams begin in mirrored formations. White attacks toward row 13 (black's goal), black attacks toward row 0 (white's goal).

---

## Movement

Movement is the foundation of Taka. How far and where you can move depends on whether you have the ball.

### Without the Ball

| Direction | Distance |
|-----------|----------|
| Toward opponent's goal | **3 squares** |
| Horizontal (sideways) | 2 squares |
| Backward (toward own goal) | 2 squares |
| Diagonal (any direction) | 2 squares |

*White's "forward" is toward row 13. Black's "forward" is toward row 0.*

### With the Ball

| Direction | Distance |
|-----------|----------|
| Any direction | **1 square** |

This creates the central trade-off: moving with the ball is slow. To progress quickly, you must pass.

### Movement Rules

1. **Path blocking:** Cannot move through squares occupied by other pieces
2. **Goal restriction:** Only goalies can enter goal areas
3. **Ball pickup:** Moving onto a square with a loose ball automatically picks it up
4. **Immediate stop:** Movement ends when landing on a square (cannot continue through)

---

## Facing Direction

Every piece faces one of four cardinal directions: north, south, east, or west.

### Why It Matters

1. **Passing cone:** You can only pass within a 90-degree cone in front of you
2. **Tackle protection:** You cannot be tackled from behind
3. **Turning costs a turn:** Changing direction uses your turn

### Turning

- You can turn to face any of the four cardinal directions
- Turning ends your turn (you cannot turn AND do something else)
- Indicated by clicking an adjacent square in the direction you want to face

---

## Passing

Passing is how you progress the ball quickly and create opportunities.

### Basic Rules

1. **90-degree cone:** You can only pass within the front-facing 90-degree cone based on your facing direction
   - Facing north: Can pass north, northeast, or northwest
   - Facing south: Can pass south, southeast, or southwest
   - Facing east: Can pass east, northeast, or southeast
   - Facing west: Can pass west, northwest, or southwest

2. **Line of sight:** The pass must follow one of the 8 directions (straight or diagonal)

3. **Adjacent blocking:** If a piece (teammate or opponent) is immediately adjacent in a direction, you cannot pass in that direction (exception: you CAN pass to an adjacent teammate)

### Pass Types

| Type | Description |
|------|-------------|
| **Ground pass** | Direct pass to a teammate in your passing cone |
| **Chip pass** | Pass over pieces that are 2+ squares away — allows passing "over" obstacles |
| **Pass to empty square** | Place the ball on an empty square for a teammate to collect |

### Consecutive Passing

After receiving a pass, you can immediately pass again without moving. This enables combination play:
- Wall passes (give-and-go)
- Third-man runs
- Quick switches of play

**Exception:** Chip passes break the consecutive passing chain.

---

## Shooting

### Shooting Zones

You can only shoot from your team's shooting zone:

| Team | Shooting zone |
|------|---------------|
| White | Rows 9-13 |
| Black | Rows 0-4 |

### How to Score

1. Be in your shooting zone
2. Have a clear line to the opponent's goal area
3. Pass the ball into the goal area (row 0 columns 3-6 for white attacking, row 13 columns 3-6 for black attacking)
4. If no goalie blocks it, **GOAL**

### Goal Area

- The goal area is the 4 squares (columns 3-6) at each end
- Only goalies can occupy the goal area
- A shot is blocked if the goalie is positioned in the path

---

## Tackling

Tackling is how you win the ball back.

### Requirements

1. **Adjacent:** Your piece must be adjacent to the opponent with the ball (including diagonals)
2. **Not from behind:** You cannot tackle an opponent from behind their facing direction

### Tackle Protection (Facing Direction)

The player with the ball is protected from tackles coming from behind:

| Target faces | Cannot be tackled from |
|--------------|------------------------|
| North | South (behind) |
| South | North (behind) |
| East | West (behind) |
| West | East (behind) |

*Tackles from the front or sides are always allowed.*

### Outcome

A successful tackle wins the ball. The tackler now has possession.

---

## Offside

Taka implements a simplified offside rule.

### Definition

A player is **offside** if:
1. They are closer to the opponent's goal than the ball, AND
2. They are closer to the opponent's goal than the second-to-last defender

### When Offside Applies

- Offside is checked when passing
- You cannot pass to a player in an offside position
- The check happens at the moment of the pass

### Example

If white is attacking (toward row 13):
- Ball is at row 8
- White player A is at row 10
- Black defenders: one at row 13 (goalie), one at row 11
- Player A is offside (ahead of ball AND ahead of second-to-last defender at row 11)

---

## Goalies

Goalies are special pieces that protect the goal.

### Goalie Properties

- Start **off the board** (unactivated)
- Are the only pieces that can enter the goal area
- Can move and pass like regular pieces when outside the goal area

### Activation

Goalies are activated when the ball enters their team's defensive zone:

**White's goalie activation zone:**
- Rows 0-3, columns 3-6 (main area)
- Row 3, columns 2 and 7 (wing positions)

**Black's goalie activation zone:**
- Rows 10-13, columns 3-6 (main area)
- Row 10, columns 2 and 7 (wing positions)

### Blocking Shots

- Position your goalie in the goal area
- If a shot's path passes through the goalie, the shot is **blocked**
- The ball goes to the goalie

---

## Turn Structure

### Your Turn

On your turn, you may do **one** of the following:

| Action | Description |
|--------|-------------|
| **Move** | Move a piece (3 forward/2 other without ball, 1 any with ball) |
| **Pass** | Pass the ball to a teammate or empty square |
| **Turn** | Change a piece's facing direction |
| **Tackle** | Tackle an adjacent opponent with the ball |
| **Shoot** | Pass the ball into the opponent's goal area |
| **Activate goalie** | Bring your goalie onto the board (when triggered) |

### Consecutive Actions

Some actions chain together:
- **Consecutive passing:** After receiving a pass, can immediately pass again
- **Ball pickup:** Moving onto a ball square automatically picks it up (part of movement)

---

## Winning

### Scoring

A goal is scored when the ball enters the opponent's goal area without being blocked by a goalie.

### Game Length

Games can be played with:
- **Timed matches:** Most goals in time limit wins
- **First to X:** First team to score X goals wins
- **Single goal:** First goal wins (quick games)

---

## Strategic Concepts

### Movement Trade-offs

The 3-forward vs 1-with-ball asymmetry creates tension:
- Fast progression requires passing
- Holding the ball gives control but limits speed
- Creating overloads requires coordinated off-ball movement

### Facing Direction Management

- Face where you want to pass next
- Face attackers you might need to shield from
- Turning costs a turn — plan ahead

### Space Creation

- Move pieces without the ball to create passing options
- Stretch the defense with diagonal runs
- Create 2v1 situations through off-ball movement

### Defensive Shape

- Maintain compact defensive positioning
- Control passing lanes through positioning
- Approach from front/sides to enable tackles

### Offside Trap

- Push defensive line up to catch attackers offside
- Risk: leaves space behind if opponents time runs correctly

---

## Physical Edition

### Components

| Item | Quantity | Purpose |
|------|----------|---------|
| Game board | 1 | 14×10 grid with pitch markings |
| White pieces | 11 | White team (including goalie) |
| Black pieces | 11 | Black team (including goalie) |
| Ball token | 1 | Indicates ball position |
| Direction indicators | 22 | Show which way pieces face |
| Rule book | 1 | Complete rules reference |
| Quick-start guide | 1 | Learn in 5 minutes |

### Piece Design

Each piece shows:
- Team color (white or black)
- Clear facing direction indicator (arrow or asymmetric shape)
- Distinguishable at a glance

---

## Online Edition

### Features

| Feature | Description |
|---------|-------------|
| **Matchmaking** | Play against opponents of similar skill |
| **ELO rating** | Skill-based ranking system |
| **Tutorial** | 16 interactive lessons covering all mechanics |
| **Async play** | Make moves on your schedule |
| **Ranked ladder** | Compete for leaderboard positions |

### Skill Levels

| Level | ELO Range | Description |
|-------|-----------|-------------|
| Beginner | Under 500 | Learning the mechanics |
| Intermediate | 500-1000 | Understanding strategy |
| Advanced | Over 1000 | High-level tactical play |

### Tutorial System

The online tutorial teaches all mechanics through interactive lessons:

1. Basic movement (without ball)
2. Turning (facing direction)
3. Movement with ball
4. Passing to teammates
5. Consecutive passing
6. Passing to empty squares
7. Ball pickup
8. Receiving passes
9. Chip passes
10. Shooting
11. Tackling
12. Tackle positioning
13. Activating goalies
14. Blocking shots

Each lesson includes:
- Interactive board state
- Step-by-step instructions
- Validation that you executed correctly
- Progress tracking

---

## Why Taka Works

### Captures Real Football

Every mechanic maps to real tactical decisions:

| Taka Mechanic | Real Football Equivalent |
|---------------|--------------------------|
| 3-forward vs 1-with-ball | Off-ball movement vs possession |
| 90-degree passing cone | Body orientation for passing |
| Facing direction | Shielding, awareness, positioning |
| Chip passes | Lofted balls over defensive lines |
| Consecutive passing | Combination play, one-touch football |
| Shooting zones | Final third attacking positions |
| Tackle from front/sides | Pressing angles, defensive approach |
| Offside rule | Defensive lines, timing runs |
| Goalie activation | Shot-stopping situations |

### Accessible Strategy

- Rules can be learned in 10 minutes
- Depth emerges from interaction of simple rules
- No football knowledge required to play
- Football knowledge improves play (tactical intuition transfers)

### Competitive Depth

- High skill ceiling (like chess)
- Every decision matters
- Multiple valid strategies
- Games decided by quality of decisions, not luck

---

## Summary

Taka is football distilled to its tactical essence. A 14×10 board. 22 pieces. Simple rules that create deep strategy.

Move without the ball (fast). Move with the ball (slow). Pass to progress. Face where you want to play. Tackle from the front. Score to win.

**The best move isn't always obvious.** That's what makes it a strategy game.
