import type { AuthenticatedRouteHandler, RouteHandler } from "@/types/index";
import prisma from "@/database";
import { z } from "zod";
import { guestSessionManager } from "@/utils/guestSession";

const getGameSchema = z.object({
  params: z.object({
    gameId: z.string().cuid(),
  }),
});

const createGuestGameSchema = z.object({
  body: z.object({
    guestUsername: z.string().min(1).max(20).optional(),
  }),
});

export const createGame: AuthenticatedRouteHandler = async (req, res) => {
  try {
    if (!req.user?.onboardingComplete) {
      res.status(400).json({ error: "Onboarding must be completed" });
      return;
    }

    const game = await prisma.game.create({
      data: {
        whitePlayerId: req.user.id,
        whitePlayerUsername: req.user.username,
        status: "waiting",
        currentTurn: "white",
      },
      include: {
        whitePlayer: {
          select: {
            id: true,
            username: true,
            elo: true,
          },
        },
        blackPlayer: {
          select: {
            id: true,
            username: true,
            elo: true,
          },
        },
      },
    });

    res.status(201).json({
      gameId: game.id,
      game: game,
    });
  } catch (error) {
    console.error("Create game error:", error);
    res.status(500).json({ error: "Failed to create game" });
  }
};

export const createGuestGame: RouteHandler = async (req, res) => {
  try {
    const { body } = createGuestGameSchema.parse(req);
    const guestUsername =
      body.guestUsername || guestSessionManager.generateGuestUsername();

    // Create guest session
    const guestSession = guestSessionManager.createSession(guestUsername);

    const game = await prisma.game.create({
      data: {
        whitePlayerGuestId: guestSession.sessionId,
        whitePlayerUsername: guestSession.username,
        status: "waiting",
        currentTurn: "white",
      },
    });

    res.status(201).json({
      gameId: game.id,
      game: game,
      guestSession: {
        sessionId: guestSession.sessionId,
        username: guestSession.username,
      },
    });
  } catch (error) {
    if (error instanceof z.ZodError) {
      res.status(400).json({ error: "Invalid request data" });
      return;
    }

    console.error("Create guest game error:", error);
    res.status(500).json({ error: "Failed to create game" });
  }
};

export const getGame: RouteHandler = async (req, res) => {
  try {
    const { params } = getGameSchema.parse(req);

    const game = await prisma.game.findUnique({
      where: { id: params.gameId },
      include: {
        whitePlayer: {
          select: {
            id: true,
            username: true,
            elo: true,
          },
        },
        blackPlayer: {
          select: {
            id: true,
            username: true,
            elo: true,
          },
        },
      },
    });

    if (!game) {
      res.status(404).json({ error: "Game not found" });
      return;
    }

    res.json({ game });
  } catch (error) {
    if (error instanceof z.ZodError) {
      res.status(400).json({ error: "Invalid game ID" });
      return;
    }

    console.error("Get game error:", error);
    res.status(500).json({ error: "Failed to get game" });
  }
};
