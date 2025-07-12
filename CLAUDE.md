# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Next.js 15 frontend application for "Taka" - a turn-based football strategy game. The app is a static landing page showcasing the game concept with a visual game board component.

## Key Commands

- `npm run dev` - Start development server
- `npm run build` - Build for production (exports static files)
- `npm run start` - Start production server
- `npm run lint` - Run ESLint

## Project Structure

```
src/
├── app/
│   ├── components/
│   │   └── StaticGameBoard.tsx     # Game board visualization component
│   ├── favicon.ico
│   ├── globals.css                 # Global Tailwind CSS styles
│   ├── layout.tsx                  # Root layout with font configuration
│   └── page.tsx                    # Main landing page
```

## Technology Stack

- **Framework**: Next.js 15 with App Router
- **Styling**: Tailwind CSS v4 with PostCSS
- **TypeScript**: Full TypeScript setup
- **Fonts**: Geist Sans and Geist Mono from Google Fonts
- **Export**: Configured for static export (`output: "export"`)

## Architecture Notes

### Game Board Component (`StaticGameBoard.tsx`)

- Renders a 10x14 grid football field
- Uses coordinate system (A-J rows, 1-14 columns)
- Displays static initial game positions for white and black teams
- Each team has 11 pieces including 1 goalie
- Pieces show color, ball possession, goalie status, and facing direction
- Grid includes goal areas, field dividers, and proper football field markings

### Landing Page (`page.tsx`)

- Marketing-focused single page with hero section, features, how-to-play, and footer
- Integrates the StaticGameBoard component as a preview
- Responsive design with Tailwind utilities
- Uses semantic HTML structure

### Styling Approach

- Tailwind-first styling with utility classes
- Green/blue color scheme matching football theme
- Responsive grid layouts
- Custom gradients and shadows

## Development Notes

- Next.js is configured for static export (no server-side features)
- Images are unoptimized for static hosting compatibility
- Trailing slashes enabled for static hosting
- ESLint configured with Next.js rules
