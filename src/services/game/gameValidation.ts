import { Piece } from "@/classes/Piece";
import { Position } from "@/classes/Position";
import {
  BOARD_COLS,
  BOARD_ROWS,
  DIRECTION_VECTORS,
  FORWARD_MOVE_DISTANCE,
  OTHER_MOVE_DISTANCE,
} from "@/utils/constants";
import { BoardType } from "@/types/types";
import {
  getAdjacentPositions,
  findBallPositions,
} from "@/services/game/boardHelpers";

/**
 * Utility functions for game validation and calculations
 */

/**
 * Checks if a piece is in an offside position
 * @param piece - The piece to check for offside
 * @param ballPosition - Position of the ball
 * @param boardLayout - Current board layout
 * @returns True if the piece is offside
 */
export const isPlayerOffside = (
  piece: Piece,
  ballPosition: Position,
  boardLayout: BoardType,
): boolean => {
  const piecePos = piece.getPositionOrThrowIfUnactivated();
  const [pieceRow] = piecePos.getPositionCoordinates();
  const [ballRow] = ballPosition.getPositionCoordinates();
  const pieceColor = piece.getColor();

  // Determine which goal the piece is attacking
  // White attacks toward black's goal at row 13 (higher row numbers)
  // Black attacks toward white's goal at row 0 (lower row numbers)
  const isWhite = pieceColor === "white";

  // Check if piece is closer to opponent's goal than the ball
  const closerToGoalThanBall = isWhite
    ? pieceRow > ballRow // For white, higher row number = closer to black's goal (row 13)
    : pieceRow < ballRow; // For black, lower row number = closer to white's goal (row 0)

  if (!closerToGoalThanBall) {
    return false; // Not ahead of ball, so not offside
  }

  // Find all opponent pieces and sort by distance to their own goal
  const opponentPieces: Piece[] = [];

  for (let row = 0; row < BOARD_ROWS; row++) {
    for (let col = 0; col < BOARD_COLS; col++) {
      const square = boardLayout[row][col];
      if (square instanceof Piece && square.getColor() !== pieceColor) {
        opponentPieces.push(square);
      }
    }
  }

  // Sort opponent pieces by distance to their goal (closest to farthest)
  opponentPieces.sort((a, b) => {
    const [aRow] = a.getPositionOrThrowIfUnactivated().getPositionCoordinates();
    const [bRow] = b.getPositionOrThrowIfUnactivated().getPositionCoordinates();

    if (isWhite) {
      // White is attacking black, so sort black pieces by distance to their own goal (row 13)
      // Higher row = closer to their goal, so descending sort
      return bRow - aRow;
    } else {
      // Black is attacking white, so sort white pieces by distance to their own goal (row 0)
      // Lower row = closer to their goal, so ascending sort
      return aRow - bRow;
    }
  });

  // Get the second-to-last opponent (second closest to their own goal)
  if (opponentPieces.length < 2) {
    return false; // If fewer than 2 opponents, cannot be offside
  }

  const secondToLastOpponent = opponentPieces[1];
  const [secondOpponentRow] = secondToLastOpponent
    .getPositionOrThrowIfUnactivated()
    .getPositionCoordinates();

  // Check if piece is closer to goal than second-to-last opponent
  if (isWhite) {
    return pieceRow > secondOpponentRow; // White piece is closer to black goal (higher row)
  } else {
    return pieceRow < secondOpponentRow; // Black piece is closer to white goal (lower row)
  }
};

/**
 * Gets all valid movement positions for a piece. Accounts for blocked paths due to other pieces
 * @param piece - The piece to get movement targets for
 * @param boardLayout - Current board layout
 * @returns Array of valid target positions
 */
export const getValidMovementTargets = (
  piece: Piece,
  boardLayout: BoardType,
): Position[] => {
  const validMoves: Position[] = [];
  const [curRow, curCol] = piece
    .getPositionOrThrowIfUnactivated()
    .getPositionCoordinates();

  // Get movement distance based on ball possession
  const hasBall = piece.getHasBall();
  const color = piece.getColor();
  const isGoalie = piece.getIsGoalie();

  // For each direction vector, trace the path step by step
  for (const [dRow, dCol] of DIRECTION_VECTORS) {
    // Calculate max distance for this direction
    let maxDistance = 0;

    if (hasBall) {
      // Ball movement: 1 square in any direction
      maxDistance = 1;
    } else {
      // Standard movement rules
      const isTowardOpponentGoal =
        (color === "white" && dRow > 0) || (color === "black" && dRow < 0);

      const isHorizontal = dCol === 0 && dRow !== 0;
      const isVertical = dRow === 0 && dCol !== 0;
      const isDiagonal = dRow !== 0 && dCol !== 0;

      if (isTowardOpponentGoal) {
        maxDistance = FORWARD_MOVE_DISTANCE;
      } else if (isHorizontal || isVertical || isDiagonal) {
        maxDistance = OTHER_MOVE_DISTANCE;
      }
    }

    // Trace the path in this direction, checking each square
    for (let distance = 1; distance <= maxDistance; distance++) {
      const newRow = curRow + dRow * distance;
      const newCol = curCol + dCol * distance;

      // Check bounds
      if (
        newRow < 0 ||
        newRow >= BOARD_ROWS ||
        newCol < 0 ||
        newCol >= BOARD_COLS
      ) {
        break;
      }

      const newPosition = new Position(newRow, newCol);
      const square = boardLayout[newRow][newCol];

      // Check if piece can enter this position
      if (square instanceof Piece) {
        // Path is blocked by another piece - cannot continue in this direction
        break;
      }

      // Check goal area restrictions (only goalies can enter goal areas)
      if (!isGoalie && newPosition.isPositionInGoal()) {
        // Path is blocked by goal area restriction - cannot continue in this direction
        break;
      }

      // Square is empty or has a ball - valid move
      if (square === null || square === "ball") {
        validMoves.push(newPosition);
      }
    }
  }

  return validMoves;
};

/**
 * Get all valid pass targets for a piece
 * @param origin - Piece to get pass targets of
 * @param boardLayout - Current board layout
 * @param checkOffside - Whether to check for offside (default: false)
 */
export const getValidPassTargets = (
  origin: Piece,
  boardLayout: BoardType,
  checkOffside: boolean = false,
): Position[] => {
  const validMoves: Position[] = [];

  const [curRow, curCol] = origin
    .getPositionOrThrowIfUnactivated()
    .getPositionCoordinates();
  const facingDirection = origin.getFacingDirection();

  // Get ball position for offside checks (only if needed)
  const ballPosition = checkOffside
    ? findBallPositions(boardLayout)[0] ||
      origin.getPositionOrThrowIfUnactivated()
    : null;

  for (const [dRow, dCol] of DIRECTION_VECTORS) {
    if (
      (facingDirection === "north" && dRow > 0) ||
      (facingDirection === "east" && dCol < 0) ||
      (facingDirection === "south" && dRow < 0) ||
      (facingDirection === "west" && dCol > 0)
    ) {
      continue;
    }

    for (let distance = 1; ; distance++) {
      const newRow = curRow + dRow * distance;
      const newCol = curCol + dCol * distance;

      if (newRow < 0 || newRow > 13 || newCol < 0 || newCol > 9) {
        break;
      }

      const newPosition = new Position(newRow, newCol);
      const square = boardLayout[newRow][newCol];

      if (square instanceof Piece) {
        if (square.getColor() === origin.getColor()) {
          // Check if the target piece is offside (only if checkOffside is true)
          if (
            !checkOffside ||
            !isPlayerOffside(square, ballPosition!, boardLayout)
          ) {
            validMoves.push(newPosition);
          }
          // Break as we can't pass behind a piece
          break;
        } else if (distance === 1) {
          // If this is an adjacent square AND the piece that is adjacent to the current piece is an opponent piece, we cannot chip pass, so break this path
          break;
        }
        // For opponent pieces at distance > 1, continue the loop to allow chip passes
      }
    }
  }

  return validMoves;
};

/**
 * Get all valid empty square pass targets. This is used for passing the ball to an empty square
 * @param origin - Piece to check pass targets for
 * @param boardLayout - Current board layout
 */
export const getValidEmptySquarePassTargets = (
  origin: Piece,
  boardLayout: BoardType,
): Position[] => {
  const validMoves: Position[] = [];

  const [curRow, curCol] = origin
    .getPositionOrThrowIfUnactivated()
    .getPositionCoordinates();
  const facingDirection = origin.getFacingDirection();

  for (const [dRow, dCol] of DIRECTION_VECTORS) {
    if (
      (facingDirection === "north" && dRow > 0) ||
      (facingDirection === "east" && dCol < 0) ||
      (facingDirection === "south" && dRow < 0) ||
      (facingDirection === "west" && dCol > 0)
    ) {
      continue;
    }

    for (let distance = 1; ; distance++) {
      const newRow = curRow + dRow * distance;
      const newCol = curCol + dCol * distance;

      if (newRow < 0 || newRow > 13 || newCol < 0 || newCol > 9) {
        break;
      }

      const newPosition = new Position(newRow, newCol);
      const square = boardLayout[newRow][newCol];

      if (square instanceof Piece) {
        // Break as we can't pass behind a piece
        break;
      }

      validMoves.push(newPosition);
    }
  }

  return validMoves;
};

/**
 * Get the valid turn targets for a piece
 * @param piece - Piece to get turn targets of
 */
export const getTurnTargets = (
  piece: Piece,
): Array<{
  position: Position;
  direction: "north" | "south" | "west" | "east";
}> => {
  const [row, col] = piece
    .getPositionOrThrowIfUnactivated()
    .getPositionCoordinates();
  const targets = [];

  if (row - 1 >= 0)
    targets.push({
      position: new Position(row - 1, col),
      direction: "north" as const,
    });
  if (row + 1 < BOARD_ROWS)
    targets.push({
      position: new Position(row + 1, col),
      direction: "south" as const,
    });
  if (col - 1 >= 0)
    targets.push({
      position: new Position(row, col - 1),
      direction: "west" as const,
    });
  if (col + 1 < BOARD_COLS)
    targets.push({
      position: new Position(row, col + 1),
      direction: "east" as const,
    });

  return targets;
};

/**
 * Checks if a position is a valid movement target for a piece
 * @param piece - The piece to validate movement for
 * @param position - The position to validate
 * @param boardLayout - Current board layout
 * @returns True if the position is a valid movement target
 */
export const isPositionValidMovementTarget = (
  piece: Piece,
  position: Position,
  boardLayout: BoardType,
): boolean => {
  const targets = getValidMovementTargets(piece, boardLayout);

  return targets.some((target) => target.equals(position));
};

/**
 * Checks if a tackler can tackle a target piece based on facing direction
 * @param tackler - The piece attempting to tackle
 * @param target - The target piece with the ball
 * @returns True if tackle is allowed, false if target is facing away
 */
export const canTackle = (tackler: Piece, target: Piece): boolean => {
  const tacklerPos = tackler.getPositionOrThrowIfUnactivated();
  const targetPos = target.getPositionOrThrowIfUnactivated();
  const targetFacing = target.getFacingDirection();

  const [tacklerRow, tacklerCol] = tacklerPos.getPositionCoordinates();
  const [targetRow, targetCol] = targetPos.getPositionCoordinates();

  // Calculate direction from target to tackler
  const rowDiff = tacklerRow - targetRow;
  const colDiff = tacklerCol - targetCol;

  // Determine if target is facing away from tackler
  switch (targetFacing) {
    case "north":
      // Target faces north, can't tackle if tackler is to the south
      return rowDiff <= 0;
    case "south":
      // Target faces south, can't tackle if tackler is to the north
      return rowDiff >= 0;
    case "west":
      // Target faces west, can't tackle if tackler is to the east
      return colDiff <= 0;
    case "east":
      // Target faces east, can't tackle if tackler is to the west
      return colDiff >= 0;
    default:
      return true;
  }
};

/**
 * Get all valid tackle targets for a piece
 * @param tackler - Piece attempting to tackle
 * @param boardLayout - Current board layout
 * @returns Array of positions with opponent pieces that can be tackled
 */
export const getValidTackleTargets = (
  tackler: Piece,
  boardLayout: BoardType,
): Position[] => {
  const validTargets: Position[] = [];
  const tacklerPos = tackler.getPositionOrThrowIfUnactivated();
  const adjacentPositions = getAdjacentPositions(tacklerPos);

  for (const pos of adjacentPositions) {
    const [row, col] = pos.getPositionCoordinates();

    // Check if position is within board bounds
    if (row < 0 || row >= BOARD_ROWS || col < 0 || col >= BOARD_COLS) {
      continue;
    }

    const square = boardLayout[row][col];

    // Must be an opponent piece with the ball
    if (
      square instanceof Piece &&
      square.getColor() !== tackler.getColor() &&
      square.getHasBall()
    ) {
      // Check if tackle is allowed (not facing away)
      if (canTackle(tackler, square)) {
        validTargets.push(pos);
      }
    }
  }

  return validTargets;
};

/**
 * Checks if a pass crosses shooting zones (full move rule)
 */
export const isCrossZonePass = (
  fromPosition: Position,
  toPosition: Position,
): boolean => {
  const [fromRow] = fromPosition.getPositionCoordinates();
  const [toRow] = toPosition.getPositionCoordinates();

  // White's shooting zone: rows 9-13, Black's shooting zone: rows 0-4
  // Middle zone: rows 5-8
  const isFromWhiteZone = fromRow >= 9;
  const isFromBlackZone = fromRow <= 4;
  const isToWhiteZone = toRow >= 9;
  const isToBlackZone = toRow <= 4;

  // Cross-zone if passing from one shooting zone to another, or from shooting zone to middle/other shooting zone
  return (
    (isFromWhiteZone && (isToBlackZone || (toRow >= 5 && toRow <= 8))) ||
    (isFromBlackZone && (isToWhiteZone || (toRow >= 5 && toRow <= 8))) ||
    (fromRow >= 5 && fromRow <= 8 && (isToWhiteZone || isToBlackZone))
  );
};
