import { Piece } from "@/classes/Piece";
import { Position } from "@/classes/Position";
import { BoardSquareType, BoardType, PlayerColor } from "@/types/types";
import { BOARD_COLS, BOARD_ROWS } from "@/utils/constants";

/**
 * Board manipulation utilities
 */

/**
 * Creates a BOARD_ROWS x BOARD_COLS blank board filled with null values
 * @returns A 2D array representing an empty game board
 */
export const createBlankBoard = (): BoardType =>
  Array.from({ length: BOARD_ROWS }, () =>
    (Array(BOARD_COLS) as (Piece | null)[]).fill(null),
  );

/**
 * Sets the board layout with the given pieces
 * @param pieces - Array of pieces to place on the board
 * @param balls - Where to place balls
 * @returns New board layout
 */
export const createBoardLayout = (
  pieces: Piece[],
  balls?: Position[],
): BoardType => {
  const boardLayout = createBlankBoard();

  pieces.forEach((piece) => {
    const pos = piece.getPosition();

    // If an unactivated goalie, skjp this piece
    if (!(pos instanceof Position)) return;

    const [row, col] = pos.getPositionCoordinates();
    boardLayout[row][col] = piece;
  });

  if (balls) {
    balls.forEach((pos) => {
      const [row, col] = pos.getPositionCoordinates();

      if (boardLayout[row][col] !== null) {
        throw new Error("Balls and pieces cannot be overlapping");
      }

      boardLayout[row][col] = "ball";
    });
  }

  return boardLayout;
};

/**
 * Get the piece or ball at a board square
 * @param position - Position to get info for
 * @param boardLayout - Current board layout
 */
export const getBoardSquare = (
  position: Position,
  boardLayout: BoardType,
): BoardSquareType => {
  const [row, col] = position.getPositionCoordinates();
  return boardLayout[row][col];
};

/**
 * Gets the piece at a specific position on the board
 * @param position - The position to check
 * @param boardLayout - Current board layout
 * @returns The piece at the position or null if empty
 */
export const getPieceAtPosition = (
  position: Position,
  boardLayout: BoardType,
): Piece | null => {
  const square = getBoardSquare(position, boardLayout);

  return square instanceof Piece ? square : null;
};

/**
 * Place a ball at a given position
 * @param position - Position to place ball
 * @param boardLayout - Current board layout
 * @returns New board layout with ball placed
 */
export const placeBallAtPosition = (
  position: Position,
  boardLayout: BoardType,
): BoardType => {
  const square = getBoardSquare(position, boardLayout);

  if (square !== null) {
    throw new Error("Cannot place a ball on an occupied square");
  }

  const [row, col] = position.getPositionCoordinates();
  const newBoardLayout = boardLayout.map((row) => [...row]);
  newBoardLayout[row][col] = "ball";

  return newBoardLayout;
};

/**
 * Move a piece to a new position on the board
 * @param piece - Piece to move
 * @param newPosition - Target position for the piece
 * @param boardLayout - Current board layout
 * @returns New board layout with piece moved
 */
export const movePieceOnBoard = (
  piece: Piece,
  newPosition: Position,
  boardLayout: BoardType,
): BoardType => {
  if (getPieceAtPosition(newPosition, boardLayout)) {
    throw new Error("There can't be a piece at the new location.");
  }

  const [oRow, oCol] = piece
    .getPositionOrThrowIfUnactivated()
    .getPositionCoordinates();
  const [nRow, nCol] = newPosition.getPositionCoordinates();

  const newBoardLayout = boardLayout.map((row) => [...row]);
  const targetSquare = newBoardLayout[nRow][nCol];

  // If there's a ball at the target position, pick it up
  if (targetSquare === "ball") {
    piece.setHasBall(true);
  }

  piece.setPosition(newPosition);
  newBoardLayout[oRow][oCol] = null;
  newBoardLayout[nRow][nCol] = piece;

  return newBoardLayout;
};

/**
 * Get all adjacent positions to a given position
 * @param position Position to check
 */
export const getAdjacentPositions = (position: Position): Position[] => {
  const [row, col] = position.getPositionCoordinates();

  return [
    new Position(row - 1, col), // north
    new Position(row + 1, col), // south
    new Position(row, col - 1), // west
    new Position(row, col + 1), // east
    new Position(row - 1, col - 1), // northwest
    new Position(row - 1, col + 1), // northeast
    new Position(row + 1, col - 1), // southwest
    new Position(row + 1, col + 1), // southeast
  ];
};

/**
 * Get all pieces of playerColor within 1 square
 * @param position Position to check
 * @param playerColor Player color to check pieces for
 * @param boardLayout Board layout to check against
 */
export const getAdjacentPieces = (
  position: Position,
  playerColor: PlayerColor,
  boardLayout: BoardType,
): Piece[] => {
  const pieces: Piece[] = [];

  const adjacentPositions = getAdjacentPositions(position);

  for (const adjPos of adjacentPositions) {
    const [adjRow, adjCol] = adjPos.getPositionCoordinates();

    // Check if position is within board bounds
    if (
      adjRow >= 0 &&
      adjRow < BOARD_ROWS &&
      adjCol >= 0 &&
      adjCol < BOARD_COLS
    ) {
      const piece = getPieceAtPosition(adjPos, boardLayout);

      if (piece?.getColor() !== playerColor) continue;

      if (piece) pieces.push(piece);
    }
  }

  return pieces;
};

/**
 * Get the position of the first ball we find
 * @param boardLayout Current board layout
 */
export const findLooseBalls = (boardLayout: BoardType): Position[] => {
  const positions: Position[] = [];

  for (let row_idx = 0; row_idx < BOARD_ROWS; row_idx++) {
    for (let col_idx = 0; col_idx < BOARD_COLS; col_idx++) {
      if (boardLayout[row_idx][col_idx] === "ball") {
        positions.push(new Position(row_idx, col_idx));
      }
    }
  }

  return positions;
};

/**
 * Find the ball position - either loose or with a piece
 * @param boardLayout - Current board layout
 * @returns Position of the ball or null if not found
 */
export const findBallPositions = (boardLayout: BoardType): Position[] => {
  // First check for loose ball
  const looseBalls = findLooseBalls(boardLayout);

  const positions = [...looseBalls];

  // Ball also might be with a piece, find which piece has it
  for (let row = 0; row < BOARD_ROWS; row++) {
    for (let col = 0; col < BOARD_COLS; col++) {
      const square = boardLayout[row][col];
      if (square instanceof Piece && square.getHasBall()) {
        positions.push(new Position(row, col));
      }
    }
  }

  return positions;
};

/**
 * Swap positions of two pieces on the board and handle tackle logic
 * @param tackler - The piece performing the tackle
 * @param target - The piece being tackled (must have ball)
 * @param boardLayout - Current board layout
 * @returns New board layout with pieces swapped and ball transferred
 */
export const swapPiecePositions = (
  tackler: Piece,
  target: Piece,
  boardLayout: BoardType,
): BoardType => {
  if (!target.getHasBall()) {
    throw new Error("Target piece must have ball to be tackled");
  }

  const tacklerPos = tackler.getPositionOrThrowIfUnactivated();
  const targetPos = target.getPositionOrThrowIfUnactivated();

  const [tacklerRow, tacklerCol] = tacklerPos.getPositionCoordinates();
  const [targetRow, targetCol] = targetPos.getPositionCoordinates();

  // Calculate vertical direction tackler moved (toward goal)
  const rowDiff = targetRow - tacklerRow;
  let newFacingDirection = tackler.getFacingDirection();

  if (rowDiff > 0) {
    // Tackler moved south (toward white's goal, black's attack direction)
    newFacingDirection = tackler.getColor() === "white" ? "north" : "south";
  } else if (rowDiff < 0) {
    // Tackler moved north (toward black's goal, white's attack direction)  
    newFacingDirection = tackler.getColor() === "white" ? "south" : "north";
  }

  // Create new board layout
  const newBoardLayout = boardLayout.map((row) => [...row]);

  // Transfer ball and update positions
  target.setHasBall(false);
  tackler.setHasBall(true);
  tackler.setFacingDirection(newFacingDirection);

  // Swap positions
  tackler.setPosition(targetPos);
  target.setPosition(tacklerPos);

  // Update board layout
  newBoardLayout[tacklerRow][tacklerCol] = target;
  newBoardLayout[targetRow][targetCol] = tackler;

  return newBoardLayout;
};

/**
 * Add a new piece to the board. This is used to activate goalies. When calling this function, make sure to check if a
 * ball is at the piece's position
 * @param piece Piece to add. Position is inferred from this piece
 * @param boardLayout Current board layout
 */
export const addPieceToBoard = (piece: Piece, boardLayout: BoardType) => {
  const pos = piece.getPositionOrThrowIfUnactivated();

  const square = getBoardSquare(pos, boardLayout);

  if (square instanceof Piece) {
    throw new Error("Trying to add a piece on top of another piece");
  }

  const [row, col] = pos.getPositionCoordinates();

  // This will overwrite any balls
  boardLayout[row][col] = piece;

  return boardLayout;
};
