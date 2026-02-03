# Taka: A Turn-Based Football Strategy Game

## Introduction

**Taka** is a sophisticated turn-based football (soccer) strategy board game that combines the tactical depth of chess with the excitement of football. Played on a 14×10 grid battlefield, two players command 11-piece teams in a battle of wits where every move, pass, and tackle must be carefully considered.

Unlike real-time football games that test reflexes, Taka rewards strategic thinking, spatial awareness, and the ability to plan multiple moves ahead. It's football distilled into its purest tactical form.

**The Twist:** Each team has their own ball. Both teams are simultaneously on offense and defense, creating a unique dual-threat dynamic where you must balance attacking with your ball while defending against your opponent's.

---

## The Playing Field

### Board Layout

The Taka board consists of a **14-row × 10-column grid** with coordinates labeled A-J (columns) and 1-14 (rows), similar to chess notation.

```
        A   B   C   D   E   F   G   H   I   J
    14  [ ] [ ] [ ] [G] [G] [G] [G] [ ] [ ] [ ]   ← Black's Goal
    13  [ ] [ ] [ ] [ ] [ ] [ ] [ ] [ ] [ ] [ ]
    12  [ ] [ ] [ ] [ ] [ ] [ ] [ ] [ ] [ ] [ ]
    ...
     3  [ ] [ ] [ ] [ ] [ ] [ ] [ ] [ ] [ ] [ ]
     2  [ ] [ ] [ ] [ ] [ ] [ ] [ ] [ ] [ ] [ ]
     1  [ ] [ ] [ ] [G] [G] [G] [G] [ ] [ ] [ ]   ← White's Goal
```

- **Goals:** Located at rows 1 and 14, spanning columns D-G (4 squares wide)
- **Goal Areas:** Protected zones that only goalies can enter
- **Shooting Zones:**
  - White must be in rows 9-13 to shoot at Black's goal
  - Black must be in rows 1-5 to shoot at White's goal

### Teams and Pieces

Each team consists of **11 pieces**:
- **10 Field Players:** Can move anywhere except goal areas
- **1 Goalie:** The only piece that can enter and defend the goal area

**Team Colors:**
- **White:** Starts at the bottom (rows 1-4), faces North
- **Black:** Starts at the top (rows 11-14), faces South

**Starting Balls:**
- Each team begins with **one ball** held by a designated piece
- White's ball starts with piece at F3
- Black's ball starts with piece at E12
- Both balls are in play simultaneously throughout the game

---

## Core Game Mechanics

### Movement System

Movement in Taka depends on two factors: the direction of travel and whether the piece has the ball.

#### Without the Ball

| Direction | Maximum Distance |
|-----------|-----------------|
| Forward (toward opponent's goal) | 3 squares |
| Horizontal (left/right) | 2 squares |
| Backward | 2 squares |

#### With the Ball

When carrying the ball, all movement is restricted to **1 square in any direction**. This creates natural tension between advancing quickly and maintaining possession.

#### Movement Restrictions
- Cannot move through other pieces (path must be clear)
- Cannot move onto a square occupied by your own piece
- Cannot enter goal areas (unless you're a goalie)

### Facing Direction

Every piece has a **facing direction** (North, South, East, or West) that determines:
- Which direction they can pass
- Which opponents they can tackle
- How vulnerable they are to being tackled

Default facing:
- White pieces face **North** (toward Black's goal)
- Black pieces face **South** (toward White's goal)

### The Turn Action

After moving or passing, you can perform a **turn action** to change your piece's facing direction. This is done by clicking one of the four adjacent squares (N/S/E/W), and the piece will face that direction.

Turning is strategically crucial for:
- Setting up passes to teammates
- Protecting against tackles
- Preparing offensive plays

---

## Ball Mechanics

### The Two-Ball System

Taka's most distinctive feature is its **two-ball system**. Each team has their own ball, meaning:

- **Simultaneous offense and defense:** You're always attacking with your ball while defending against your opponent's
- **No possesion battles:** You don't fight over the same ball—you fight to score with yours first
- **Parallel threats:** Every turn involves decisions about advancing your attack AND blocking theirs
- **Tackling steals the opponent's ball:** When you tackle, you take their ball from them

This creates a fundamentally different dynamic than traditional single-ball football. You can't simply "park the bus" on defense—you must advance your own ball to win.

### Ball Possession

- A piece **automatically picks up** a ball when it moves onto the ball's square
- Only **one piece can hold a ball** at a time (but two different pieces can each hold their team's ball)
- Balls can exist **independently** on the field (loose balls)

### Passing System

Passing is the heart of Taka's tactical gameplay. There are two types of passes:

#### Pass to Teammate

**Cone Restriction:** You can only pass within a **90-degree cone** in the direction your piece is facing.

```
          Facing North
              ↑
            /   \
           /     \
          /       \
         [  PASS   ]
         [  ZONE   ]

            [PIECE]
```

**Chip Passing:** Passes can go **over opponent pieces** (like a chip in football), but:
- Cannot chip over pieces **directly adjacent** to the passer
- Cannot pass through your own pieces

#### Pass to Empty Square

You can pass to any empty square within your facing cone. This is useful for:
- Positioning the ball strategically
- Setting up plays
- Moving the ball when teammates aren't available

**Shooting Zone Restriction:** To pass into the goal area, your piece must be in the designated shooting zone.

### Consecutive Passing

One of Taka's most exciting mechanics: you can make **up to 2 passes per turn**. This enables:
- Quick one-two plays
- Fast breaks through the defense
- Complex combination attacks

After making a pass, you can immediately select the receiving piece and pass again (if valid).

---

## Tackling

Tackles are how you steal the opponent's ball and disrupt their attack.

### Tackle Rules

1. **Proximity:** Can only tackle **adjacent** opponent pieces (one square away)
2. **Facing Restriction:** **Cannot tackle** an opponent who is facing away from you
3. **Result:** On successful tackle:
   - The tackler and target **swap positions**
   - The tackler **steals the opponent's ball**
   - The tackler's facing direction changes toward the tackled piece's original position

### Tackle Strategy

In a two-ball game, tackling is primarily **defensive**—you're stopping the opponent's attack by stealing their ball:
- Position pieces to face away from dangerous opponents (protect your ball carrier)
- Set up tackles by maneuvering to an opponent's "blind side"
- Use the position swap to penetrate defensive lines
- A successful tackle can completely derail an opponent's offensive push

---

## The Offside Rule

Taka enforces a simplified **offside rule**:

A piece is **offside** if:
1. It is **closer to the opponent's goal** than the ball
2. AND closer to the opponent's goal than the **second-to-last opponent piece**

Offside pieces **cannot receive passes**. This prevents cherry-picking and encourages proper build-up play.

---

## Goalkeeping

### Goalie Privileges

- **Only goalies** can enter and occupy goal area squares
- Goalies can **block shots** (similar to tackling, but for incoming balls)
- Goalies are the last line of defense

### Goalie Activation

Goalies start **unactivated** and must be deployed onto the field:
- White's goalie activates from designated zones near row 1
- Black's goalie activates from designated zones near row 14

This adds another strategic layer: when to commit your goalie to the field.

---

## Scoring

A goal is scored when:
1. A piece with a ball **enters the opponent's goal area**, OR
2. A ball is **passed into the opponent's goal area**

**First to score wins.** Since both teams have a ball, it's a race—whoever gets their ball into the opponent's goal first wins the game. This creates constant tension: do you commit resources to attack faster, or hold back to defend?

---

## Turn Structure

A complete turn in Taka follows this sequence:

```
1. SELECT    →  Choose one of your pieces
2. MOVE      →  Move to a valid position (optional if passing)
3. TURN      →  Change facing direction (optional)
4. PASS      →  Pass to teammate or empty square (optional)
5. PASS #2   →  Make a second consecutive pass (optional)
6. END TURN  →  Control passes to opponent
```

Or, alternatively:

```
1. SELECT    →  Choose one of your pieces
2. TACKLE    →  Tackle an adjacent opponent
3. TURN      →  Change facing direction (optional)
4. END TURN  →  Control passes to opponent
```

---

## Game Modes

### Tutorial Mode

Taka features a **comprehensive 16-step interactive tutorial** that teaches:

| Step | Lesson |
|------|--------|
| 1-2 | Basic piece movement |
| 3-4 | Movement with ball, turning |
| 5-6 | Basic passing, consecutive passing |
| 7-9 | Passing to empty squares, ball pickup |
| 10 | Chip passing over opponents |
| 11-14 | Shooting, goalie activation, blocking |
| 15-16 | Tackling mechanics |

Each tutorial step features a pre-configured board state that isolates the mechanic being taught.

### Multiplayer Mode

**Real-time competitive matches** with:
- Live synchronization via WebSockets
- Backend move validation (no cheating)
- ELO-based matchmaking

**Skill Levels:**
| Level | Starting ELO |
|-------|-------------|
| Beginner | 200 |
| Intermediate | 500 |
| Advanced | 1000 |

### Guest Play

Jump into games without creating an account. Guest players can:
- Play against registered users
- Have persistent sessions
- Experience full gameplay

---

## Strategic Concepts

### The Two-Ball Dilemma

The fundamental strategic question in Taka: **attack or defend?**

With two balls in play, you must constantly balance:
- **Pushing your attack:** Moving your ball toward the opponent's goal
- **Slowing their attack:** Positioning pieces to tackle or block their ball carrier
- **Tempo control:** Sometimes it's better to consolidate position than rush forward

The best players find ways to do both—advancing their ball while simultaneously threatening the opponent's ball carrier.

### Controlling Space

Like real football, controlling the midfield and key passing lanes is crucial. Use your facing directions to create passing networks that opponents can't easily intercept.

### The One-Two

With consecutive passing, master the one-two: pass to a teammate, have them pass back (or to another teammate), and penetrate defensive lines.

### Defensive Shape

Keep pieces facing toward likely attack vectors. A piece facing away from an attacker is vulnerable to tackles and position swaps.

### Ball Carrier Dilemma

The ball carrier can only move 1 square, creating natural decision points:
- Move slowly with the ball for control
- Pass to advance quickly but risk interception
- Draw defenders then release to open teammates

### Shooting Zone Management

You must be in the shooting zone to score. This creates natural bottlenecks and makes the final approach the most contested area of the board.

---

## Technical Architecture

### Frontend
- **Next.js 15** with App Router
- **TypeScript** for type safety
- **Tailwind CSS** for styling
- **Zustand** for state management
- **Socket.IO** client for real-time communication

### Backend
- **Bun** runtime (faster than Node.js)
- **Express.js 5** framework
- **PostgreSQL** with **Prisma ORM**
- **Socket.IO** for WebSocket handling
- **JWT** authentication with magic links

### Game Logic
- **Piece class:** Encapsulates movement rules and state
- **Position class:** Handles coordinate transformations
- **Validation service:** 560+ lines of rule enforcement
- **Board helpers:** Immutable state transformations

---

## What Makes Taka Unique

1. **Two-Ball System:** Each team has their own ball—simultaneous offense and defense creates a race to score first
2. **Turn-Based Football:** Removes reflexes from the equation, pure tactical thinking
3. **Facing Direction System:** Adds a chess-like element of controlling attack angles
4. **Consecutive Passing:** Enables combination play rarely seen in board games
5. **Realistic Rules:** Offside, tackling restrictions, and shooting zones mirror real football
6. **Accessible Yet Deep:** Easy to learn movement, lifetime to master positioning
7. **Real-Time Multiplayer:** Competitive play with skill-based matchmaking
8. **Passwordless Auth:** Frictionless access via magic links

---

## Conclusion

Taka transforms football into a thoughtful, strategic experience where every piece's position and facing direction matters. The two-ball system creates a constant tension between attack and defense—you can never fully commit to one without risking the other.

Whether you're racing your ball toward the goal, tackling to steal your opponent's ball, or setting up a devastating consecutive passing sequence, Taka rewards the player who can balance aggression with caution and see three moves ahead on both ends of the field.

It's football as a game of perfect information—no luck, no reflexes, just pure strategy and the eternal question: attack or defend?

---

*Welcome to Taka. Two balls. One goal. First to score wins.*
