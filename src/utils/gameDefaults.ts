import type { GamePiece, BallPosition, GameState } from "@/types/index";

export function createDefaultGameState(
  whitePlayerId: string,
  blackPlayerId: string,
): GameState {
  // Default piece positions for standard formation (0-indexed coordinates)
  // White team: F3 (x:5,y:2 with ball), E3 (x:4,y:2), C4 (x:2,y:3), H4 (x:7,y:3), D5 (x:3,y:4), G5 (x:6,y:4), B6 (x:1,y:5), I6 (x:8,y:5), E7 (x:4,y:6), F7 (x:5,y:6), plus unactivated goalie
  // Black team: E12 (x:4,y:11), F12 (x:5,y:11), H11 (x:7,y:10), C11 (x:2,y:10), G10 (x:6,y:9), D10 (x:3,y:9), I9 (x:8,y:8), B9 (x:1,y:8), F8 (x:5,y:7), E8 (x:4,y:7), plus unactivated goalie
  const whitePieces: GamePiece[] = [
    { id: "w1", playerId: whitePlayerId, x: 5, y: 2, type: "player", hasBall: true, facingDirection: "south" },
    { id: "w2", playerId: whitePlayerId, x: 4, y: 2, type: "player" },
    { id: "w3", playerId: whitePlayerId, x: 2, y: 3, type: "player" },
    { id: "w4", playerId: whitePlayerId, x: 7, y: 3, type: "player" },
    { id: "w5", playerId: whitePlayerId, x: 3, y: 4, type: "player" },
    { id: "w6", playerId: whitePlayerId, x: 6, y: 4, type: "player" },
    { id: "w7", playerId: whitePlayerId, x: 1, y: 5, type: "player" },
    { id: "w8", playerId: whitePlayerId, x: 8, y: 5, type: "player" },
    { id: "w9", playerId: whitePlayerId, x: 4, y: 6, type: "player" },
    { id: "w10", playerId: whitePlayerId, x: 5, y: 6, type: "player" },
    { id: "wgoalie", playerId: whitePlayerId, type: "goalie" },
  ];

  const blackPieces: GamePiece[] = [
    { id: "b1", playerId: blackPlayerId, x: 4, y: 11, type: "player", hasBall: true, facingDirection: "north" },
    { id: "b2", playerId: blackPlayerId, x: 5, y: 11, type: "player" },
    { id: "b3", playerId: blackPlayerId, x: 7, y: 10, type: "player" },
    { id: "b4", playerId: blackPlayerId, x: 2, y: 10, type: "player" },
    { id: "b5", playerId: blackPlayerId, x: 6, y: 9, type: "player" },
    { id: "b6", playerId: blackPlayerId, x: 3, y: 9, type: "player" },
    { id: "b7", playerId: blackPlayerId, x: 8, y: 8, type: "player" },
    { id: "b8", playerId: blackPlayerId, x: 1, y: 8, type: "player" },
    { id: "b9", playerId: blackPlayerId, x: 5, y: 7, type: "player" },
    { id: "b10", playerId: blackPlayerId, x: 4, y: 7, type: "player" },
    { id: "bgoalie", playerId: blackPlayerId, type: "goalie" },
  ];

  // No loose balls - pieces have them
  const ballPositions: BallPosition[] = [];

  return {
    pieces: [...whitePieces, ...blackPieces],
    ballPositions: ballPositions,
  };
}

export const BOARD_CONFIG = {
  width: 10,  // 10 columns (A-J)
  height: 14, // 14 rows (1-14)
  whiteGoal: [
    { x: 3, y: 0 }, // D1 (0-indexed)
    { x: 4, y: 0 }, // E1
    { x: 5, y: 0 }, // F1
    { x: 6, y: 0 }, // G1
  ],
  blackGoal: [
    { x: 3, y: 13 }, // D14 (0-indexed)
    { x: 4, y: 13 }, // E14
    { x: 5, y: 13 }, // F14
    { x: 6, y: 13 }, // G14
  ],
};

export function isGoalScored(
  ballPositions: BallPosition[],
  pieces?: GamePiece[],
): "white" | "black" | null {
  // Check loose balls in goal areas
  for (const ball of ballPositions) {
    // Check if ball is in white's goal (row 0, 0-indexed)
    if (ball.y === 0 && ball.x >= 3 && ball.x <= 6) {
      return "black"; // Black scores in white's goal
    }
    // Check if ball is in black's goal (row 13, 0-indexed)
    if (ball.y === 13 && ball.x >= 3 && ball.x <= 6) {
      return "white"; // White scores in black's goal
    }
  }

  // Check pieces with balls in goal areas
  if (pieces) {
    for (const piece of pieces) {
      if (piece.hasBall && piece.x !== undefined && piece.y !== undefined) {
        // Check if piece with ball is in white's goal (row 0, 0-indexed)
        if (piece.y === 0 && piece.x >= 3 && piece.x <= 6) {
          // Determine which team scored based on piece ID
          const isWhitePiece = piece.id.toLowerCase().startsWith('w');
          return isWhitePiece ? "white" : "black"; // Piece's team scores
        }
        // Check if piece with ball is in black's goal (row 13, 0-indexed)
        if (piece.y === 13 && piece.x >= 3 && piece.x <= 6) {
          // Determine which team scored based on piece ID
          const isWhitePiece = piece.id.toLowerCase().startsWith('w');
          return isWhitePiece ? "white" : "black"; // Piece's team scores
        }
      }
    }
  }

  return null;
}
