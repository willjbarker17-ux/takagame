import jwt from "jsonwebtoken";
import config from "@/config";
import prisma from "@/database";
import type { AuthenticatedSocket } from "@/types/index";
import { logger } from "@/utils/logger";
import { guestSessionManager } from "@/utils/guestSession";

interface JwtPayload {
  sub: string;
  type: string;
}

export async function socketAuthMiddleware(
  socket: AuthenticatedSocket,
  next: (err?: Error) => void,
) {
  try {
    const token =
      (socket.handshake.auth?.token as string) ||
      (socket.handshake.headers?.authorization as string)?.replace(
        "Bearer ",
        "",
      );

    const guestSessionId = socket.handshake.auth?.guestSessionId as string;
    const guestUsername = socket.handshake.auth?.guestUsername as string;

    // Try JWT authentication first
    if (token) {
      try {
        const decoded = jwt.verify(token, config.JWT_SECRET) as JwtPayload;

        if (decoded.type !== "access") {
          logger.warn("Socket connection denied - invalid token type", {
            socketId: socket.id,
            ip: socket.handshake.address,
          });
          return next(new Error("Invalid token type"));
        }

        const user = await prisma.user.findUnique({
          where: { id: decoded.sub },
        });

        if (!user || !user.verified) {
          logger.warn(
            "Socket connection denied - user not found or not verified",
            {
              socketId: socket.id,
              ip: socket.handshake.address,
              userId: decoded.sub,
            },
          );
          return next(new Error("User not found or not verified"));
        }

        socket.user = user;
        logger.socket("Authenticated user connected", {
          socketId: socket.id,
          userId: user.id,
          ip: socket.handshake.address,
          metadata: { username: user.username, elo: user.elo },
        });
        return next();
      } catch {
        logger.warn("JWT authentication failed, trying guest auth", {
          socketId: socket.id,
          ip: socket.handshake.address,
        });
        // Fall through to guest authentication
      }
    }

    // Try guest session authentication
    if (guestSessionId) {
      console.log("Backend validating guest session ID:", guestSessionId);
      const guestSession = guestSessionManager.getSession(guestSessionId);

      if (!guestSession) {
        console.log("Guest session validation failed - session not found");
        logger.warn("Socket connection denied - invalid guest session", {
          socketId: socket.id,
          ip: socket.handshake.address,
          guestSessionId,
        });
        return next(new Error("Invalid or expired guest session"));
      }

      console.log("Guest session validated successfully:", guestSession.sessionId);

      // Update socket ID in session
      guestSessionManager.updateSocketId(guestSessionId, socket.id);
      socket.guestUser = guestSession;

      logger.socket("Guest user connected", {
        socketId: socket.id,
        ip: socket.handshake.address,
        metadata: {
          guestUsername: guestSession.username,
          sessionId: guestSession.sessionId,
        },
      });
      return next();
    }

    // Create new guest session if username provided
    if (guestUsername && guestUsername.trim().length > 0) {
      const trimmedUsername = guestUsername.trim().slice(0, 20); // Limit length
      const guestSession = guestSessionManager.createSession(trimmedUsername);
      guestSessionManager.updateSocketId(guestSession.sessionId, socket.id);
      socket.guestUser = guestSession;

      logger.socket("New guest user connected", {
        socketId: socket.id,
        ip: socket.handshake.address,
        metadata: {
          guestUsername: guestSession.username,
          sessionId: guestSession.sessionId,
        },
      });
      return next();
    }

    // No valid authentication method provided
    logger.warn("Socket connection denied - no valid authentication", {
      socketId: socket.id,
      ip: socket.handshake.address,
    });
    return next(
      new Error(
        "Authentication required: provide JWT token, guest session, or guest username",
      ),
    );
  } catch (error) {
    logger.error("Socket authentication error", error as Error, {
      socketId: socket.id,
      ip: socket.handshake.address,
    });
    next(new Error("Authentication failed"));
  }
}
