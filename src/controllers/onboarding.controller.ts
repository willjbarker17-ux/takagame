import { z } from "zod";
import type { RouteHandler } from "@/types";
import prisma from "@/database";

const SKILL_LEVELS = ["beginner", "intermediate", "advanced"] as const;

const ELO_MAP: Record<(typeof SKILL_LEVELS)[number], number> = {
  beginner: 200,
  intermediate: 500,
  advanced: 1000,
};

const skillLevelEnum = z.enum(SKILL_LEVELS);

const completeOnboardingSchema = z.object({
  skillLevel: skillLevelEnum,
  username: z
    .string()
    .min(3, "Username is required")
    .max(32)
    .regex(/^[a-zA-Z0-9_]+$/, "Username must be alphanumeric or underscore"),
});

/**
 * Complete the onboarding process for a user.
 */
export const completeOnboarding: RouteHandler = async (req, res, next) => {
  try {
    const user = await prisma.user.findUnique({
      where: { id: req.user!.userId },
    });

    if (!user) {
      res.status(404).json({ error: "User not found." });
      return;
    }

    if (user.onboardingComplete) {
      res.status(403).json({ error: "Onboarding has already been completed." });
      return;
    }

    const { skillLevel, username } = completeOnboardingSchema.parse(req.body);

    // Check if username is already taken
    const existingUserWithUsername = await prisma.user.findUnique({
      where: { username },
    });

    if (existingUserWithUsername && existingUserWithUsername.id !== user.id) {
      res.status(409).json({ error: "This username is already taken." });
      return;
    }

    const elo = ELO_MAP[skillLevel];

    const updatedUser = await prisma.user.update({
      where: { id: req.user!.userId },
      data: {
        username,
        elo,
        onboardingComplete: true,
      },
    });

    res.json(updatedUser);
  } catch (error) {
    next(error);
  }
};
