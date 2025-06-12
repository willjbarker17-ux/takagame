import { z } from "zod";
import type { RouteHandler } from "../types";
import { prisma } from "../database";

const skillLevelEnum = z.enum(["beginner", "intermediate", "advanced"]);

const completeOnboardingSchema = z.object({
  skillLevel: skillLevelEnum,
});

const ELO_MAP = {
  beginner: 200,
  intermediate: 500,
  advanced: 1000,
};

/**
 * Complete the onboarding process for a user.
 */
export const completeOnboarding: RouteHandler = async (req, res, next) => {
  try {
    const user = await prisma.user.findUnique({ where: { id: req.user!.userId } });

    if (!user) {
        res.status(404).json({ error: "User not found." });
        return;
    }

    if (user.onboardingComplete) {
      res.status(403).json({ error: "Onboarding has already been completed." });
      return;
    }

    const { skillLevel } = completeOnboardingSchema.parse(req.body);

    const elo = ELO_MAP[skillLevel];

    const updatedUser = await prisma.user.update({
      where: { id: req.user!.userId },
      data: {
        elo,
        onboardingComplete: true,
      },
    });

    res.json(updatedUser);
  } catch (error) {
    next(error);
  }
}; 