# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Next.js 15 frontend application for "Taka" - a turn-based football strategy game. The app includes a marketing landing page, an interactive tutorial system, and a rules page. The core feature is an interactive game board with tutorial progression and state management.

## Key Commands

- `npm run dev` - Start development server on port 8000
- `npm run build` - Build for production (exports static files)
- `npm run start` - Start production server on port 8000
- `npm run lint` - Run ESLint
- `npm run format` - Format code with Prettier
- `npm run format:check` - Check code formatting with Prettier

## Project Structure

```
src/
├── app/
│   ├── components/
│   │   ├── BoardCell.tsx           # Individual board cell component
│   │   ├── Header.tsx              # Common header component
│   │   ├── Piece.tsx               # Individual game piece component
│   │   ├── SoccerBallIcon.tsx      # Soccer ball SVG component
│   │   ├── StaticGameBoard.tsx     # Static game board for landing page
│   │   ├── TutorialGameBoard.tsx   # Interactive tutorial game board
│   │   └── TutorialPanel.tsx       # Tutorial step instructions panel
│   ├── rules/
│   │   └── page.tsx                # Game rules page
│   ├── tutorial/
│   │   └── page.tsx                # Interactive tutorial page
│   ├── favicon.ico
│   ├── globals.css                 # Global Tailwind CSS styles
│   ├── layout.tsx                  # Root layout with font configuration
│   └── page.tsx                    # Main landing page
├── classes/
│   ├── Piece.ts                    # Game piece class with movement logic
│   └── Position.ts                 # Board position class with coordinates
├── hooks/
│   └── useTutorialStore.ts         # Zustand store for tutorial state
├── services/
│   ├── boardHelpers.ts             # Board manipulation and utility functions
│   └── gameValidation.ts           # Game logic validation functions
├── types/
│   └── types.ts                    # TypeScript type definitions
└── utils/
    └── constants.ts                # Game constants and configuration
```

## Technology Stack

- **Framework**: Next.js 15 with App Router
- **Styling**: Tailwind CSS v4 with PostCSS
- **TypeScript**: Full TypeScript setup
- **State Management**: Zustand for tutorial state
- **Icons**: Lucide React for UI icons
- **Fonts**: Geist Sans and Geist Mono from Google Fonts
- **Export**: Configured for static export (`output: "export"`)

## Architecture Notes

### Interactive Tutorial System

The app features a comprehensive tutorial system built with Zustand for state management:

- **Tutorial Store** (`useTutorialStore.ts`): Manages tutorial progression, board state, piece selection, and game interactions
- **Step Progression**: Eleven tutorial steps from welcome to completion, each with predefined board states
- **Interactive Board** (`TutorialGameBoard.tsx`): Handles click events, piece selection, movement validation, and ball passing
- **Tutorial Panel** (`TutorialPanel.tsx`): Displays step-by-step instructions and progress tracking

### Game Logic Classes

- **Piece Class** (`classes/Piece.ts`): Handles piece movement rules, ball possession, facing direction, and movement validation
  - Two movement patterns: standard movement (3 forward, 2 horizontal/backward) vs ball movement (1 square in any direction)
  - Tracks color, position, ball possession, and facing direction
  - Built-in goal area restrictions for non-goalie pieces
- **Position Class** (`classes/Position.ts`): Manages board coordinates and position-based logic
  - 14x10 coordinate system (rows 0-13, columns 0-9)
  - Goal detection logic for positions (0,3-6) and (13,3-6)
  - Immutable position objects with equality checking

### Service Layer

- **Board Helpers** (`services/boardHelpers.ts`): Core board manipulation functions
  - Board creation, piece placement, and layout management
  - Ball placement and movement utilities
  - Adjacent position/piece calculations
  - Board state transformations with immutable updates
- **Game Validation** (`services/gameValidation.ts`): Movement and action validation
  - Movement target calculation with path blocking
  - Pass target validation with line-of-sight and facing direction rules
  - Turn target generation for direction changes
  - Empty square pass target calculation

### Game Board Components

#### Static Game Board (`StaticGameBoard.tsx`)

- Renders a 14x10 grid football field for the landing page
- Uses coordinate system (rows 0-13, columns 0-9)
- Displays static initial game positions for white and black teams
- Each team has 11 pieces including 1 goalie

#### Interactive Tutorial Board (`TutorialGameBoard.tsx`)

- Reactive game board that responds to tutorial state changes
- Handles piece selection, movement targets, pass targets, and turn indicators
- Real-time visual feedback for valid moves and interactions
- Integrated with tutorial step progression logic
- Uses `BoardCell` components for individual squares with click handling

### Page Structure

#### Landing Page (`page.tsx`)

- Marketing-focused single page with hero section, features, how-to-play, and footer
- Integrates the StaticGameBoard component as a preview
- Responsive design with Tailwind utilities

#### Tutorial Page (`tutorial/page.tsx`)

- Interactive learning environment with side-by-side board and instructions
- Full-screen responsive layout with tutorial progression
- Combines TutorialGameBoard and TutorialPanel components

### Styling Approach

- Tailwind-first styling with utility classes
- Green/blue color scheme matching football theme
- Responsive grid layouts
- Custom gradients and shadows

### State Management Patterns

- **Zustand Store**: Central store pattern for tutorial state with reactive updates
- **Immutable Updates**: Board layout updates create new arrays to trigger React re-renders
- **Action Functions**: Exported functions handle complex state transitions (piece movement, ball passing, step progression)
- **Computed State**: Functions like `getSquareInfo()` derive display state from base state
- **Event Handling**: `handleSquareClick()` manages all board interactions with comprehensive state validation

### Key Game Mechanics

- **Movement Validation**: Pieces have different movement rules based on ball possession and facing direction
- **Ball Passing**: Supports passing to pieces and empty squares with line-of-sight validation
- **Tutorial Steps**: Predefined states for each tutorial step with automatic progression triggers
- **Turn System**: Direction selection mode for piece rotation with visual indicators
- **Consecutive Passing**: Advanced mechanic allowing chained passes in tutorial

## Development Notes

- Next.js is configured for static export (no server-side features)
- Images are unoptimized for static hosting compatibility
- Trailing slashes enabled for static hosting
- ESLint configured with Next.js rules
- Development server runs on port 8000 (both dev and start commands)
- Prettier integration for consistent code formatting

## Important Development Guidelines

### Board Coordinate System

- The game uses a 14x10 grid (14 rows, 10 columns)
- Rows are indexed 0-13 (representing positions 1-14 in game terms)
- Columns are indexed 0-9 (representing positions A-J in game terms)
- Goal areas are at rows 0 and 13, columns 3-6

### Tutorial Step Flow

The tutorial follows this progression:

1. `welcome` - Introduction
2. `basic_movement` - Basic piece movement without ball
3. `turning` - Direction changes for pieces with ball
4. `movement_with_ball` - Movement when possessing the ball
5. `passing` - Basic passing to other pieces
6. `consecutive_pass` - Chained passing sequences
7. `ball_empty_square` - Passing to empty squares
8. `ball_pickup` - Moving to pick up balls from empty squares
9. `receiving_passes` - Receiving passes from empty squares
10. `chip_pass` - Advanced passing over opponents
11. `shooting` - Scoring goals
12. `tackling` - Tackling opponents
13. `tackling_positioning` - Tactical positioning before tackling
14. `activating_goalies` - Activating goalie pieces
15. `blocking_shots` - Using goalies to block shots
16. `completed` - Tutorial completion

### State Management Patterns

- All board state updates must create new arrays/objects for React reactivity
- Use the service layer functions (`boardHelpers.ts`, `gameValidation.ts`) for game logic
- The Zustand store manages both UI state and game state
- Tutorial progression is handled through predefined state configurations

### Game Rules Implementation

- Pieces have different movement patterns based on ball possession
- Pass validation includes facing direction (90-degree cone) and line-of-sight
- Ball pickup happens automatically when moving to a square containing a ball
- Goal areas have special restrictions for non-goalie pieces

## Warnings and Recommendations

- **Never run `npm run build`** - This command is not recommended for this project's development workflow
