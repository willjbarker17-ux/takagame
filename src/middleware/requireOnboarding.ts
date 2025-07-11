import type { AuthenticatedRouteHandler } from "@/types";

/**
 * Middleware to ensure user has completed onboarding before accessing protected routes
 * This has to be used after requireUser middleware
 */
const requireOnboarding: AuthenticatedRouteHandler = (req, res, next) => {
  // User should already be authenticated and loaded via requireUser middleware
  if (!req.user) throw new Error("req.user should exist, but doesn't");

  if (!req.user.onboardingComplete) {
    res.status(403).json({
      error: "Onboarding must be completed before accessing this resource",
      needsOnboarding: true,
    });
    return;
  }

  next();
};

export default requireOnboarding;
