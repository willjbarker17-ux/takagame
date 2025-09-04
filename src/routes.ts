import type { Express, Request, Response } from "express";
import {
  emailRequestOtp,
  emailVerifyOtp,
  logout,
  refreshToken,
} from "@/controllers/auth.controller";
import { completeOnboarding } from "@/controllers/onboarding.controller";
import {
  createGame,
  createGuestGame,
  getGame,
} from "@/controllers/game.controller";
import requireUser from "@/middleware/requireUser";
import requireOnboarding from "@/middleware/requireOnboarding";
import { asyncHandler } from "@/utils/asyncHandler";

const routes = (app: Express) => {
  app.get("/", (_req: Request, res: Response) => {
    res.sendStatus(200);
  });

  // Auth routes
  app.post("/auth/magic-link", asyncHandler(emailRequestOtp));
  app.post("/auth/verify", asyncHandler(emailVerifyOtp));
  app.get("/auth/logout", asyncHandler(logout));
  app.post("/auth/refresh_token", asyncHandler(refreshToken));

  // Onboarding routes (only require authentication, not completed onboarding)
  app.post(
    "/onboarding/complete",
    requireUser,
    asyncHandler(completeOnboarding),
  );

  // Game routes
  app.post(
    "/games/create",
    requireUser,
    requireOnboarding,
    asyncHandler(createGame),
  ); // Authenticated users
  app.post("/games/create-guest", asyncHandler(createGuestGame)); // Guest users
  app.get("/games/:gameId", asyncHandler(getGame)); // Public endpoint

  app.get("/*splat", (_req: Request, res: Response) => {
    res.status(404).send();
  });
};

export default routes;
