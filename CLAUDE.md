# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

**Start development server:**

```bash
bun run dev
```

**Start production server:**

```bash
bun run start
```

**Linting:**

```bash
bun run lint          # Check for lint errors
```

**Code formatting:**

```bash
bun run format        # Format code with Prettier
```

**Testing:**

```bash
bun run test              # Run tests once
vitest                    # Run tests in watch mode
vitest run --coverage     # Run tests with coverage report
vitest run path/to/test   # Run a specific test file
```

**Database operations:**

```bash
npx prisma migrate dev    # Run database migrations
npx prisma generate       # Generate Prisma client
npx prisma studio         # Open Prisma Studio
```

## Architecture Overview

This is a **Node.js Express API** built with **TypeScript** and **Bun** runtime, designed as a soccer/sports application backend.

### Core Stack

- **Runtime**: Bun (not Node.js)
- **Framework**: Express.js v5
- **Database**: PostgreSQL with Prisma ORM
- **Authentication**: JWT with magic link system
- **Email**: Nodemailer with Gmail
- **Validation**: Zod schemas
- **Monitoring**: Sentry (production)

### Authentication Architecture

The app uses a **passwordless magic link system** that handles both login and signup in a unified flow:

1. **Magic Link Request** (`POST /auth/magic-link`) - Works for both new/existing users
2. **Email Verification** (`POST /auth/verify`) - Returns JWT + onboarding status
3. **Onboarding** (`POST /onboarding/complete`) - Sets username and ELO rating

**Key Implementation Details:**

- Users are created automatically during verification if they don't exist
- JWT tokens use both access tokens (short-lived) and refresh tokens (stored in DB)
- Refresh tokens are stored as HTTP-only cookies
- Onboarding is required for new users (sets username + skill level → ELO rating)

### Database Schema

- **User**: email, username, verified, elo, onboardingComplete
- **RefreshToken**: JWT refresh tokens with user association
- **VerificationToken**: Magic link codes with expiration

### Project Structure

- **`src/controllers/`** - Route handlers (auth, onboarding)
- **`src/middleware/`** - Express middleware (auth, error handling, validation)
- **`src/utils/`** - Utilities (JWT tokens, email sending)
- **`src/config.ts`** - Environment configuration with Zod validation
- **`src/routes.ts`** - Route definitions
- **`src/database.ts`** - Prisma client singleton

### Key Middleware

- **`requireUser`** - JWT authentication middleware
- **`requireOnboarding`** - Ensures user has completed onboarding (use after requireUser)
- **`zodErrorHandler`** - Transforms Zod validation errors
- **`errorHandler`** - Global error handler
- **`noCache`** - Prevents caching

### Environment Configuration

Configuration is validated with Zod schemas in `src/config.ts`. Required variables:

- `DATABASE_URL`, `JWT_SECRET`, `SITE_URL`, `BACKEND_URL`
- Production: `SENTRY_DSN`, `GMAIL_USER`, `GMAIL_PASSWORD`
- Development: `EMAIL_STUB_ENABLED` (console logging)

### Important Notes

- **No password authentication** - magic links only
- **Onboarding enforcement** - Users cannot access game functionality until onboarding is complete
- **ELO ratings** - Users get initial ELO based on skill level (beginner: 200, intermediate: 500, advanced: 1000)
- **Refresh token security** - Tokens are invalidated on reuse detection
- **CORS** - Configured for multiple frontend origins
- **Error handling** - Comprehensive error middleware chain

### Authorization Levels

1. **Public** - No auth required (`/auth/magic-link`)
2. **Authenticated** - JWT required (`/auth/verify`, `/onboarding/complete`)
3. **Onboarded** - Completed onboarding required (all game features, `/profile`)

## Testing Infrastructure

**Framework**: Vitest (not Jest) for better TypeScript and ESM support.

**Test Structure:**

- `tests/unit/` - Unit tests for individual functions/modules
- `tests/integration/` - Integration tests for API endpoints
- `tests/fixtures/` - Test data fixtures
- `tests/mocks/` - Mock implementations (Prisma, external services)
- `tests/helpers/` - Test utilities and helpers

**Key Testing Utilities:**

- `tests/setup.ts` - Global test setup with mocks
- `tests/helpers/testApp.ts` - Express app factory for testing
- `tests/helpers/jwt.ts` - JWT token helpers
- `tests/mocks/prisma.ts` - Prisma client mocks

**Coverage**: Configured with 80% thresholds for branches, functions, lines, and statements.

**Testing Guidelines:**

- When testing, be very intelligent with the tests you write. Make sure the tests are actually useful, and you aren't just testing random stuff

## Package Management

- Use context7 when installing packages to get their latest docs.
- This project uses Bun as the package manager and runtime.

## Important Project Context

This is a **soccer board game app backend** that allows users to play against other users. Users get matched with players with similar skills (ELOs). **NEVER add admin routes.**

## Coding Guidelines

- Never use the any type
- Always run format and lint after changes are complete to make sure your code is up to style
- NEVER disable linting rules to get around lint errors
- Never bypass linter rules. Always fix the root cause

# important-instruction-reminders

Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (\*.md) or README files. Only create documentation files if explicitly requested by the User.
