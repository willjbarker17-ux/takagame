import type { AuthenticatedSocket, GameState } from "@/types/index";
import { logger } from "@/utils/logger";
import prisma from "@/database";
import { createDefaultGameState, isGoalScored } from "@/utils/gameDefaults";

/**
 * Handle joining a game room
 */
export async function handleJoinGame(
  socket: AuthenticatedSocket,
  gameId: string,
) {
  const connectionContext = {
    socketId: socket.id,
    userId: socket.user?.id || socket.guestUser?.sessionId,
    ip: socket.handshake.address,
    userAgent: socket.handshake.headers["user-agent"],
  };

  try {
    // All users can join games (guests and registered users)
    const currentPlayer = socket.user || socket.guestUser;
    if (!currentPlayer) {
      socket.emit("error", { message: "Authentication required" });
      return;
    }

    if (!gameId) {
      socket.emit("error", { message: "Game ID is required" });
      return;
    }

    // Verify the game exists
    const game = await prisma.game.findUnique({
      where: { id: gameId },
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
      socket.emit("error", { message: "Game not found" });
      return;
    }

    // Get player identifiers
    const isRegisteredUser = !!socket.user;
    const playerId = socket.user?.id || socket.guestUser?.sessionId || "";
    const playerUsername =
      socket.user?.username || socket.guestUser?.username || "";

    // Check if player is already in the game
    const isWhitePlayer =
      game.whitePlayerId === playerId || game.whitePlayerGuestId === playerId;

    // If game is waiting and user is not the white player, make them the black player
    if (game.status === "waiting" && !isWhitePlayer) {
      const hasBlackPlayer = game.blackPlayerId || game.blackPlayerGuestId;

      if (!hasBlackPlayer) {
        // Get both player IDs for creating default game state
        const whitePlayerId =
          game.whitePlayerId || game.whitePlayerGuestId || "";
        const blackPlayerId = playerId;

        // Create default game state with pieces positioned
        const defaultGameState = createDefaultGameState(
          whitePlayerId,
          blackPlayerId,
        );

        const updateData = isRegisteredUser
          ? {
              blackPlayerId: playerId,
              blackPlayerUsername: playerUsername,
              status: "active",
              currentTurn: "white",
              gameState: defaultGameState,
              ballPositions: defaultGameState.ballPositions,
            }
          : {
              blackPlayerGuestId: playerId,
              blackPlayerUsername: playerUsername,
              status: "active",
              currentTurn: "white",
              gameState: defaultGameState,
              ballPositions: defaultGameState.ballPositions,
            };

        await prisma.game.update({
          where: { id: gameId },
          data: updateData,
        });

        // Update game object for response
        if (isRegisteredUser) {
          game.blackPlayerId = playerId;
        } else {
          game.blackPlayerGuestId = playerId;
        }
        game.blackPlayerUsername = playerUsername;
        game.status = "active";
        game.currentTurn = "white";
        game.gameState = defaultGameState;
      }
    }

    // Check if user is part of this game (now supports both registered and guest users)
    const finalIsWhitePlayer =
      game.whitePlayerId === playerId || game.whitePlayerGuestId === playerId;
    const finalIsBlackPlayer =
      game.blackPlayerId === playerId || game.blackPlayerGuestId === playerId;

    console.log("Game access check:", {
      gameId: game.id,
      playerId,
      playerUsername,
      game: {
        whitePlayerId: game.whitePlayerId,
        whitePlayerGuestId: game.whitePlayerGuestId,
        blackPlayerId: game.blackPlayerId,
        blackPlayerGuestId: game.blackPlayerGuestId,
      },
      finalIsWhitePlayer,
      finalIsBlackPlayer,
    });

    if (!finalIsWhitePlayer && !finalIsBlackPlayer) {
      console.log("ACCESS DENIED: Player not found in game");
      logger.warn("Unauthorized game websocket access attempt", {
        ...connectionContext,
        metadata: {
          gameId: game.id,
          username: playerUsername,
          playerId: playerId,
        },
      });
      socket.emit("error", { message: "Access denied" });
      return;
    }

    // Join the game room
    await socket.join(`game:${gameId}`);

    logger.socket("User joined game room", {
      ...connectionContext,
      metadata: {
        gameId: game.id,
        username: playerUsername,
        gameStatus: game.status,
        isGuest: !isRegisteredUser,
      },
    });

    // Send game state to the joining player
    socket.emit("game-joined", {
      message: `Connected to game: ${gameId}`,
      gameId: gameId,
      game: game,
    });

    // Notify other players in the room and broadcast updated game state
    socket.to(`game:${gameId}`).emit("player-joined", {
      playerId: playerId,
      username: playerUsername,
      game: game,
    });
  } catch (error) {
    logger.error("Join game error", error as Error, {
      ...connectionContext,
      metadata: { gameId },
    });
    socket.emit("error", { message: "Failed to join game" });
  }
}

export async function handleMakeMove(
  socket: AuthenticatedSocket,
  data: { gameId: string; gameState: GameState },
) {
  const connectionContext = {
    socketId: socket.id,
    userId: socket.user?.id || socket.guestUser?.sessionId,
    ip: socket.handshake.address,
    userAgent: socket.handshake.headers["user-agent"],
  };

  try {
    // All users can make moves (guests and registered users)
    const currentPlayer = socket.user || socket.guestUser;
    if (!currentPlayer) {
      socket.emit("error", { message: "Authentication required" });
      return;
    }

    const { gameId, gameState } = data;

    if (!gameId || !gameState) {
      socket.emit("error", { message: "Game ID and game state are required" });
      return;
    }

    const game = await prisma.game.findUnique({
      where: { id: gameId },
    });

    if (!game) {
      socket.emit("error", { message: "Game not found" });
      return;
    }

    // Get player identifiers
    const playerId = socket.user?.id || socket.guestUser?.sessionId || "";
    const playerUsername =
      socket.user?.username || socket.guestUser?.username || "";

    // Check if user is part of this game
    const isWhitePlayer =
      game.whitePlayerId === playerId || game.whitePlayerGuestId === playerId;
    const isBlackPlayer =
      game.blackPlayerId === playerId || game.blackPlayerGuestId === playerId;

    if (!isWhitePlayer && !isBlackPlayer) {
      socket.emit("error", { message: "Access denied" });
      return;
    }

    // Check if it's the player's turn
    const expectedTurn = isWhitePlayer ? "white" : "black";

    if (game.currentTurn !== expectedTurn) {
      socket.emit("error", { message: "Not your turn" });
      return;
    }

    // Goal detection using proper board layout
    const goalScorer = isGoalScored(gameState.ballPositions, gameState.pieces);
    const isGoal = goalScorer !== null;

    const nextTurn = isWhitePlayer ? "black" : "white";
    const status = isGoal ? "completed" : "active";
    const winner = goalScorer; // goalScorer already indicates which player scored

    // Update the game in the database
    const updatedGame = await prisma.game.update({
      where: { id: gameId },
      data: {
        gameState: gameState,
        ballPositions: gameState.ballPositions,
        currentTurn: isGoal ? null : nextTurn,
        status: status,
        winner: winner,
      },
      include: {
        whitePlayer: {
          select: { id: true, username: true, elo: true },
        },
        blackPlayer: {
          select: { id: true, username: true, elo: true },
        },
      },
    });

    logger.socket("Move made", {
      ...connectionContext,
      metadata: {
        gameId: game.id,
        username: playerUsername,
        turn: expectedTurn,
        isGoal,
      },
    });

    // Broadcast the updated game state to all players in the room
    socket.to(`game:${gameId}`).emit("game-state-updated", {
      game: updatedGame,
      gameState: gameState,
      move: {
        playerId: playerId,
        username: playerUsername,
      },
    });

    // Send confirmation to the player who made the move
    socket.emit("move-confirmed", {
      game: updatedGame,
      gameState: gameState,
    });

    // If game is over, broadcast game-over event
    if (isGoal) {
      const winnerIsCurrentPlayer = winner === expectedTurn;
      const actualWinnerUsername = winnerIsCurrentPlayer
        ? playerUsername
        : "Opponent";

      socket.to(`game:${gameId}`).emit("game-over", {
        winner: winner,
        winnerUsername: actualWinnerUsername,
        game: updatedGame,
      });

      socket.emit("game-over", {
        winner: winner,
        winnerUsername: actualWinnerUsername,
        game: updatedGame,
      });
    }
  } catch (error) {
    logger.error("Make move error", error as Error, {
      ...connectionContext,
      metadata: { gameId: data.gameId },
    });
    socket.emit("error", { message: "Failed to make move" });
  }
}
