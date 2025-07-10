# soccer-app

To install dependencies:

```bash
bun install
```

To run:

```bash
bun run dev
```

This project was created using `bun init` in bun v1.2.12. [Bun](https://bun.sh) is a fast all-in-one JavaScript runtime.

## Authentication Flow

The app uses a magic link authentication system that works for both login and signup:

### 1. Request Magic Link

- **Endpoint**: `POST /auth/magic-link`
- **Body**: `{ "email": "user@example.com" }`
- **Description**: Send a magic link to the user's email. Works for both new and existing users.

### 2. Verify Magic Link

- **Endpoint**: `POST /auth/verify`
- **Body**: `{ "code": "verification_code_from_email" }`
- **Response**:
  ```json
  {
    "accessToken": "jwt_token",
    "needsOnboarding": true|false
  }
  ```
- **Description**: Verify the magic link code. Returns an access token and indicates whether the user needs to complete onboarding.

### 3. Complete Onboarding (if needed)

- **Endpoint**: `POST /onboarding/complete`
- **Headers**: `Authorization: Bearer <access_token>`
- **Body**:
  ```json
  {
    "username": "chosen_username",
    "skillLevel": "beginner|intermediate|advanced"
  }
  ```
- **Description**: Complete the onboarding process by setting username and skill level. This determines the user's initial ELO rating:
  - `beginner`: 200 ELO
  - `intermediate`: 500 ELO
  - `advanced`: 1000 ELO

### Flow Summary

1. User enters email and clicks "Login/Signup"
2. System sends magic link to email
3. User clicks link in email, gets redirected to website
4. Frontend calls `/auth/verify` with the code from the URL
5. If `needsOnboarding` is `true`, redirect to onboarding flow
6. If `needsOnboarding` is `false`, user is fully authenticated and logged in
7. During onboarding, user sets username and skill level
8. After onboarding completion, user gets full session access

### Protected Endpoints

Once authenticated, users must complete onboarding before accessing any game functionality:

- `GET /profile` - Get user profile (requires completed onboarding)

### Other Auth Endpoints

- `GET /auth/logout` - Log out user
- `POST /auth/refresh_token` - Refresh access token

## Authorization Levels

The app has three levels of access:

1. **Public** - No authentication required (magic link request)
2. **Authenticated** - Valid JWT required (magic link verification, onboarding completion)
3. **Onboarded** - Completed onboarding required (all game functionality, profile access)

Users cannot access game features until onboarding is complete. The system enforces this through middleware that checks the `onboardingComplete` flag.
