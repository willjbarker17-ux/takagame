import type { Express, Request, Response } from "express";
import {
  emailRequestOtp,
  emailVerifyOtp,
  logout,
  refreshToken,
} from "@/controllers/auth.controller";
import { completeOnboarding } from "@/controllers/onboarding.controller";
import requireUser from "@/middleware/requireUser";

const routes = (app: Express) => {
  app.get("/", (_req: Request, res: Response) => {
    res.sendStatus(200);
  });

  // Auth routes
  app.post("/auth/magic-link", emailRequestOtp);
  app.post("/auth/verify", emailVerifyOtp);
  app.get("/auth/logout", logout);
  app.post("/auth/refresh_token", refreshToken);

  // Onboarding routes (only require authentication, not completed onboarding)
  app.post("/onboarding/complete", requireUser, completeOnboarding);

  app.get("/*splat", (_req: Request, res: Response) => {
    res.status(404).send();
  });
};

export default routes;
