import type { RouteHandler } from "@/types";
import prisma from "@/database";

/**
 * Middleware to ensure user has completed onboarding before accessing protected routes
 * This should be used after requireUser middleware
 */
const requireOnboarding: RouteHandler = async (req, res, next) => {
  // User should already be authenticated via requireUser middleware
  if (!req.user) {
    res.status(401).json({ error: "Authentication required" });
    return;
  }

  try {
    // Fetch the user to check onboarding status
    const user = await prisma.user.findUnique({
      where: { id: req.user.userId },
      select: { onboardingComplete: true },
    });

    if (!user) {
      res.status(404).json({ error: "User not found" });
      return;
    }

    if (!user.onboardingComplete) {
      res.status(403).json({
        error: "Onboarding must be completed before accessing this resource",
        needsOnboarding: true,
      });
      return;
    }

    next();
  } catch (error) {
    next(error);
  }
};

export default requireOnboarding;
