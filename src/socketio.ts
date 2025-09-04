import { Server } from "socket.io";
import type { Server as HttpServer } from "http";
import config from "@/config";
import { socketAuthMiddleware } from "@/middleware/socketAuth";
import type { AuthenticatedSocket, GameState } from "@/types/index";
import { logger } from "@/utils/logger";
import { handleJoinGame, handleMakeMove } from "@/handlers/gameHandlers";

export function initializeSocketIO(server: HttpServer): Server {
  const io = new Server(server, {
    cors: {
      origin: [config.SITE_URL, "http://localhost:8000"],
      methods: ["GET", "POST"],
      credentials: true,
    },
  });

  // Socket.IO authentication middleware
  io.use((socket, next) => {
    socketAuthMiddleware(socket as AuthenticatedSocket, next).catch(
      (error: Error) => {
        next(error);
      },
    );
  });

  // Socket.IO connection handler
  io.on("connection", (socket: AuthenticatedSocket) => {
    const connectionContext = {
      socketId: socket.id,
      userId: socket.user?.id || socket.guestUser?.sessionId,
      ip: socket.handshake.address,
      userAgent: socket.handshake.headers["user-agent"],
    };

    logger.socket("User connected", {
      ...connectionContext,
      metadata: {
        email: socket.user?.email,
        needsOnboarding: !socket.user?.onboardingComplete,
        elo: socket.user?.elo,
        isGuest: !!socket.guestUser,
        guestUsername: socket.guestUser?.username,
      },
    });

    // Send guest session information to client if this is a guest user
    if (socket.guestUser) {
      socket.emit("guest-session-created", {
        sessionId: socket.guestUser.sessionId,
        username: socket.guestUser.username,
        createdAt: socket.guestUser.createdAt,
        expiresAt: socket.guestUser.expiresAt,
      });
    }

    // Handle joining a game room
    socket.on("join-game", async (gameId: string) => {
      await handleJoinGame(socket, gameId);
    });

    // Handle making a move
    socket.on(
      "make-move",
      async (data: { gameId: string; gameState: GameState }) => {
        await handleMakeMove(socket, data);
      },
    );

    socket.on("disconnect", (reason) => {
      logger.socket("User disconnected", {
        ...connectionContext,
        metadata: {
          reason,
          email: socket.user?.email,
        },
      });
    });
  });

  return io;
}
