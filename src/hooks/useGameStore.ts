"use client";

/**
 * This file is responsible for all game logic and state management. This file provides access to the game board
 * through the useGameBoard function to give the game board access to certain functions. This state is only used for
 * an active game.
 */

import { Piece } from "@/classes/Piece";
import {
  BoardType,
  FacingDirection,
  PlayerColor,
  SquareInfoType,
} from "@/types/types";
import { create } from "zustand";
import {
  addPieceToBoard,
  createBoardLayout,
  getBoardSquare,
  getPieceAtPosition,
  movePieceOnBoard,
} from "@/services/game/boardHelpers";
import { Position } from "@/classes/Position";
import {
  getTurnTargets,
  isPositionValidMovementTarget,
} from "@/services/game/gameValidation";

interface GameState {
  /** Current game id **/
  gameId: string;
  /** Array of all pieces currently on the board. This includes unactivated goalies */
  pieces: Piece[];
  /** Array of ball positions on the board. */
  balls: Position[];
  /** 2D array representing the board layout with pieces, balls or null for empty squares.
   * This is a slave state to the pieces and balls, with pieces and balls being the master */
  boardLayout: BoardType;
  /** The unactivated white goalie piece to show at intersection. This is a slave state to the pieces array */
  whiteUnactivatedGoaliePiece: Piece | null;
  /** The unactivated black goalie piece to show at intersection. This is a slave state to the pieces array */
  blackUnactivatedGoaliePiece: Piece | null;
  /** Position of the currently selected piece, if any */
  selectedPiece: Piece | null;
  /** The user's player color **/
  playerColor: PlayerColor;
  /** The current color of the turn */
  playerTurn: PlayerColor;
  /** Show direction arrows **/
  showDirectionArrows: boolean;
  /** Should we lock the selection to the selected piece? */
  isSelectionLocked: boolean;
}

const piece1 = new Piece({
  id: "WG",
  color: "white",
  position: "white_unactivated",
  hasBall: false,
  isGoalie: true,
});

const piece2 = new Piece({
  id: "BG",
  color: "black",
  position: "black_unactivated",
  hasBall: false,
  isGoalie: true,
});

const piece3 = new Piece({
  id: "W1",
  color: "white",
  position: new Position(2, 2),
  hasBall: true,
});

const piece4 = new Piece({
  id: "B1",
  color: "black",
  position: new Position(11, 2),
  hasBall: true,
});

const piece5 = new Piece({
  id: "B2",
  color: "black",
  position: new Position(11, 4),
  hasBall: false,
});

const useGameStore = create<GameState>(() => ({
  gameId: "",
  pieces: [piece1, piece2, piece3, piece4, piece5],
  balls: [new Position(12, 6)],
  boardLayout: createBoardLayout(
    [piece1, piece2, piece3, piece4, piece5],
    [new Position(12, 6)],
  ),
  selectedPiece: null,
  playerColor: "black",
  playerTurn: "black",
  whiteUnactivatedGoaliePiece: piece1,
  blackUnactivatedGoaliePiece: piece2,
  showDirectionArrows: false,
  isSelectionLocked: false,
}));

/**
 * Handle the turn button click
 */
export const handleTurnPieceButtonClick = () => {
  const { selectedPiece } = useGameStore.getState();

  if (!selectedPiece) return;

  useGameStore.setState({
    showDirectionArrows: true,
  });
};

export const handleArrowKeyTurn = (direction: FacingDirection) => {
  const { selectedPiece } = useGameStore.getState();

  if (!selectedPiece || !selectedPiece.getHasBall()) {
    return;
  }

  turnPiece(selectedPiece, direction);
};

export const handleUnactivatedGoalieClick = (color: PlayerColor) => {
  const {
    whiteUnactivatedGoaliePiece,
    blackUnactivatedGoaliePiece,
    playerColor,
  } = useGameStore.getState();

  if (color !== playerColor) return;

  if (color === "white") {
    if (!whiteUnactivatedGoaliePiece) {
      throw new Error(
        "Attempting to activate white goalie piece, but white goalie is not activated.",
      );
    }

    useGameStore.setState({ selectedPiece: whiteUnactivatedGoaliePiece });
  } else if (color === "black") {
    if (!blackUnactivatedGoaliePiece) {
      throw new Error(
        "Attempting to black white goalie piece, but black goalie is not activated.",
      );
    }

    useGameStore.setState({ selectedPiece: blackUnactivatedGoaliePiece });
  }
};

export const getSquareInfo = (position: Position): SquareInfoType => {
  const state = useGameStore.getState();

  // No action can be taken when it's not the players turn
  if (state.playerTurn !== state.playerColor) {
    return { visual: "nothing", clickable: false };
  }

  const pieceAtPosition = getPieceAtPosition(position, state.boardLayout);

  // If we are waiting for a direction selection, the only actions are to turn the piece, transfer selection, or deselect the piece
  if (state.showDirectionArrows) {
    if (!state.selectedPiece) {
      throw new Error("Arrows are shown, but there isn't a selected piece");
    }

    const turnTargets = getTurnTargets(state.selectedPiece);

    const isSquareTurnTarget = turnTargets.some((e) =>
      e.position.equals(position),
    );

    if (isSquareTurnTarget) return { visual: "turn_target", clickable: true };

    // Selection transfer
    if (pieceAtPosition?.getColor() === state.playerColor)
      return { visual: "piece", clickable: true };

    return { visual: "nothing", clickable: false };
  }

  const playersGoalie = getUnactivatedGoalie();

  // If unactivated white goalie is selected, show movement targets in goal area
  if (state.selectedPiece && state.selectedPiece === playersGoalie) {
    const isGoalActivationTarget = state.selectedPiece
      .getGoalieActivationTargets()
      .some((e) => e.equals(position));

    // Only allow activation if the position is a valid target AND there's no piece there
    if (isGoalActivationTarget && !pieceAtPosition) {
      return { visual: "movement", clickable: true };
    }

    // Selection transfer
    if (pieceAtPosition?.getColor() === state.playerColor)
      return { visual: "piece", clickable: true };

    return { visual: "nothing", clickable: false };
  }

  if (pieceAtPosition && pieceAtPosition.getColor() === state.playerColor) {
    return { visual: "piece", clickable: true };
  }

  // Check if this square is a valid movement target for the selected piece
  if (state.selectedPiece) {
    const isMovementTarget = isPositionValidMovementTarget(
      state.selectedPiece,
      position,
      state.boardLayout,
    );

    if (isMovementTarget) {
      return { visual: "movement", clickable: true };
    }
  }

  return { visual: "nothing", clickable: false };
};

/**
 * Is it the current players turn?
 */
export const isCurrentPlayersTurn = (): boolean => {
  const { playerColor, playerTurn } = useGameStore.getState();

  return playerColor === playerTurn;
};

export const handleSquareClick = (position: Position): void => {
  // If it's not our turn, we can't do anything
  if (!isCurrentPlayersTurn()) return;

  const squareType = getSquareInfo(position);

  console.log("Square click", squareType);

  // Treat not clickable squares as deselection targets
  if (!squareType.clickable) {
    return handleBlankSquareClick();
  }

  // Route to appropriate handler based on square type
  switch (squareType.visual) {
    case "movement":
      return handleMovementClick(position);
    case "piece":
      return handlePieceClick(position);
    case "turn_target":
      return handleTurnTargetClick(position);
    default:
      return handleBlankSquareClick();
  }
};

/**
 * This is used to set the currently selected piece. This is not the function that is hit for passes.
 * @param position
 */
const handlePieceClick = (position: Position): void => {
  const { boardLayout, playerColor, isSelectionLocked } =
    useGameStore.getState();

  // Prevent the selection from changing
  if (isSelectionLocked) return;

  const pieceAtPosition = getPieceAtPosition(position, boardLayout);

  if (!pieceAtPosition || pieceAtPosition.getColor() !== playerColor) {
    return;
  }

  useGameStore.setState({
    selectedPiece: pieceAtPosition,
  });
};

const getUnactivatedGoalie = () => {
  const {
    playerColor,
    whiteUnactivatedGoaliePiece,
    blackUnactivatedGoaliePiece,
  } = useGameStore.getState();

  return playerColor === "white"
    ? whiteUnactivatedGoaliePiece
    : blackUnactivatedGoaliePiece;
};

const handleMovementClick = (position: Position): void => {
  const { selectedPiece } = useGameStore.getState();

  if (!selectedPiece) {
    throw new Error(
      "Attempting to move a piece, but there isn't a selected piece",
    );
  }

  // Pieces with the ball are not allowed to move by click
  if (selectedPiece.getHasBall()) {
    throw new Error("Attempting to move a piece with the ball by clicking.");
  }

  // If the selected piece is an unactivated goalie, activate it
  const unactivatedGoalie = getUnactivatedGoalie();
  if (selectedPiece === unactivatedGoalie) {
    activateGoalie(unactivatedGoalie, position);
    return;
  }

  movePiece(selectedPiece, position);
};

const movePiece = (piece: Piece, position: Position): void => {
  const { boardLayout } = useGameStore.getState();

  const boardSquare = getBoardSquare(position, boardLayout);
  const isPickingUpBall = boardSquare === "ball";

  useGameStore.setState({
    boardLayout: movePieceOnBoard(piece, position, boardLayout),
  });

  // If we are picking up a ball, force the user to select a direction
  if (isPickingUpBall) {
    useGameStore.setState({
      showDirectionArrows: true,
      isSelectionLocked: true,
    });
  } else {
    deselectPiece();
  }
};

/**
 * Remove an unactivated goalie
 * @param color Color of goalie to deactivate
 */
const removeUnactivatedGoalie = (color: PlayerColor): void => {
  if (color === "white") {
    useGameStore.setState({
      whiteUnactivatedGoaliePiece: null,
    });
  } else if (color === "black") {
    useGameStore.setState({
      blackUnactivatedGoaliePiece: null,
    });
  }
};

/**
 * Activate an unactivated goalie
 * @param goalie Goalie piece to activate
 * @param position Position to activate the goalie on
 */
const activateGoalie = (goalie: Piece, position: Position): void => {
  const { boardLayout, playerColor } = useGameStore.getState();

  goalie.setPosition(position);

  // Check if a ball is where the goalie is going to activate
  const square = getBoardSquare(position, boardLayout);
  const isPickingUpBall = square === "ball";

  if (isPickingUpBall) {
    goalie.setHasBall(true);
  }

  const newBoard = addPieceToBoard(goalie, boardLayout);

  useGameStore.setState({
    boardLayout: newBoard,
  });

  removeUnactivatedGoalie(playerColor);

  if (isPickingUpBall) {
    useGameStore.setState({
      isSelectionLocked: true,
      showDirectionArrows: true,
    });
  } else {
    deselectPiece();
  }
};

const handleTurnTargetClick = (position: Position): void => {
  const { selectedPiece } = useGameStore.getState();

  if (!selectedPiece) {
    throw new Error(
      "Awaiting direction selection, but there is no selected piece.",
    );
  }

  const turnTarget = getTurnTargets(selectedPiece).find((e) =>
    e.position.equals(position),
  );

  if (!turnTarget) {
    throw new Error(
      "Handle turn target click was called, but this position isn't a turn target.",
    );
  }

  turnPiece(selectedPiece, turnTarget.direction);
};

const turnPiece = (piece: Piece, direction: FacingDirection): void => {
  piece.setFacingDirection(direction);

  useGameStore.setState({
    showDirectionArrows: false,
    selectedPiece: null,
  });
};

const handleBlankSquareClick = (): void => {
  const { selectedPiece } = useGameStore.getState();

  if (!selectedPiece) return;

  // TODO: Block scenarios where we can't deselect the piece

  deselectPiece();
};

/**
 * Deselect the currently select
 */
const deselectPiece = () => {
  const { isSelectionLocked } = useGameStore.getState();

  if (isSelectionLocked) return;

  useGameStore.setState({
    selectedPiece: null,
    showDirectionArrows: false,
  });
};

/**
 * Hook to access the game store and initialize it with server data
 */
export const useGameBoard = () => {
  return useGameStore();
};
