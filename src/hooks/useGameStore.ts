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
  placeBallAtPosition,
  swapPiecePositions,
} from "@/services/game/boardHelpers";
import { Position } from "@/classes/Position";
import {
  getRelativeDirectionBetweenPositions,
  getTurnTargets,
  getValidEmptySquarePassTargets,
  getValidPassTargets,
  getValidTackleTargets,
  isCrossZonePass,
  isPassChipPass,
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
  /** Did we just make a pass and are awaiting a second? */
  awaitingConsecutivePass: boolean;
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
  awaitingConsecutivePass: false,
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

  endTurn();
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

    return { visual: "nothing", clickable: false };
  }

  // Check to see if this position is a valid pass target
  if (
    pieceAtPosition &&
    state.selectedPiece &&
    state.selectedPiece.getHasBall()
  ) {
    const passTargets = getValidPassTargets(
      state.selectedPiece,
      state.boardLayout,
      true,
    );

    const isThisPositionPassTarget = passTargets.find((p) =>
      p.equals(position),
    );

    if (isThisPositionPassTarget)
      return { visual: "pass_target", clickable: true };
  }

  // Transfer selection handler
  if (
    pieceAtPosition &&
    pieceAtPosition.getColor() === state.playerColor &&
    !state.isSelectionLocked
  ) {
    return { visual: "piece", clickable: true };
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

    return { visual: "nothing", clickable: false };
  }

  // Check if this square is a valid movement target for the selected piece
  if (
    state.selectedPiece &&
    !state.selectedPiece.getHasBall() &&
    !state.awaitingConsecutivePass
  ) {
    const isMovementTarget = isPositionValidMovementTarget(
      state.selectedPiece,
      position,
      state.boardLayout,
    );

    if (isMovementTarget) {
      return { visual: "movement", clickable: true };
    }
  }

  // Check if this position is a tackle target
  if (
    pieceAtPosition &&
    state.selectedPiece &&
    !state.selectedPiece.getHasBall() &&
    pieceAtPosition.getColor() !== state.playerColor &&
    !state.awaitingConsecutivePass
  ) {
    const tackleTargets = getValidTackleTargets(
      state.selectedPiece,
      state.boardLayout,
    );

    const positionIsTackleTarget = tackleTargets.find((p) =>
      p.equals(position),
    );

    if (positionIsTackleTarget)
      return { visual: "tackle_target", clickable: true };
  }

  // Check if this position is an empty square pass target
  if (state.selectedPiece && state.selectedPiece.getHasBall()) {
    const isValidEmptySquarePassTarget = getValidEmptySquarePassTargets(
      state.selectedPiece,
      state.boardLayout,
    ).find((p) => p.equals(position));

    if (isValidEmptySquarePassTarget)
      return { visual: "empty_pass_target", clickable: true };
  }

  return { visual: "nothing", clickable: false };
};

const endTurn = (): void => {
  useGameStore.setState((state) => {
    return {
      playerTurn: state.playerTurn === "white" ? "black" : "white",
      playerColor: state.playerColor === "white" ? "black" : "white",
      selectedPiece: null,
      isSelectionLocked: false,
      awaitingConsecutivePass: false,
    };
  });
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
    case "pass_target":
      return handlePassTargetClick(position);
    case "empty_pass_target":
      return handleEmptySquarePassTargetClick(position);
    case "tackle_target":
      return handleTackleTargetClick(position);
    default:
      return handleBlankSquareClick();
  }
};

const handleTackleTargetClick = (position: Position): void => {
  const { selectedPiece, boardLayout } = useGameStore.getState();

  if (!selectedPiece) {
    throw new Error("Trying to tackle, but there isn't a selected piece");
  }

  const targetPiece = getPieceAtPosition(position, boardLayout);

  if (!targetPiece || !targetPiece.getHasBall()) {
    throw new Error("Trying to tackle a piece that doesn't have a ball");
  }

  const newBoard = swapPiecePositions(selectedPiece, targetPiece, boardLayout);

  useGameStore.setState({
    boardLayout: newBoard,
    isSelectionLocked: true,
  });
};

const handleEmptySquarePassTargetClick = (position: Position): void => {
  const { selectedPiece } = useGameStore.getState();

  if (!selectedPiece) {
    throw new Error(
      "Trying to pass the ball to an empty square, but there is no selected piece",
    );
  }

  passBall(selectedPiece.getPositionOrThrowIfUnactivated(), position);

  deselectPiece();

  // TODO: Receiving pass

  endTurn();
};

/**
 * Pass the ball
 * @param origin Origin piece to pass ball from
 * @param destination Destination to pass ball to
 */
const passBall = (origin: Position, destination: Position) => {
  const { boardLayout } = useGameStore.getState();

  const originPiece = getPieceAtPosition(origin, boardLayout);
  const destinationPiece = getPieceAtPosition(destination, boardLayout);

  if (!originPiece) {
    throw new Error("There must be a piece at both the origin");
  }

  if (!originPiece.getHasBall()) {
    throw new Error(
      "Trying to pass a ball from a piece that doesn't currently have a ball",
    );
  }

  originPiece.setHasBall(false);

  if (destinationPiece) {
    destinationPiece.setHasBall(true);

    // Figure out what direction to make the destination piece face
    const directionToFace = getRelativeDirectionBetweenPositions(
      origin,
      destination,
    );

    destinationPiece.setFacingDirection(directionToFace);

    // Push an update to the state
    useGameStore.setState({});
  } else {
    const newBoard = placeBallAtPosition(destination, boardLayout);

    useGameStore.setState({
      boardLayout: newBoard,
    });
  }
};

const handlePassTargetClick = (position: Position): void => {
  const { selectedPiece, boardLayout, awaitingConsecutivePass } =
    useGameStore.getState();
  const pieceAtPosition = getPieceAtPosition(position, boardLayout);

  if (!selectedPiece) {
    throw new Error("Trying to pass but there isn't a selected piece");
  }

  const fromPosition = selectedPiece.getPositionOrThrowIfUnactivated();
  passBall(fromPosition, position);

  // Select the piece that received the pass
  useGameStore.setState({
    selectedPiece: pieceAtPosition,
  });

  if (
    isCrossZonePass(fromPosition, position) ||
    isPassChipPass(selectedPiece, position) ||
    awaitingConsecutivePass
  ) {
    /** If we made a cross-zone pass, second pass in consecutive pass, or a chip pass, we can now only choose direction,
    no more passes */
    useGameStore.setState({
      showDirectionArrows: true,
      isSelectionLocked: true,
    });
    return;
  }

  useGameStore.setState({
    awaitingConsecutivePass: true,
    isSelectionLocked: true,
  });
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
    const pickedUpBall = activateGoalie(unactivatedGoalie, position);

    if (!pickedUpBall) {
      endTurn();
    }

    return;
  }

  const pickedUpBall = movePiece(selectedPiece, position);

  if (!pickedUpBall) {
    endTurn();
  }
};

/**
 * Move a piece to a position, and return if we picked up the ball
 * @param piece Piece to move
 * @param position Position to move piece to
 * @returns Did we pick up a loose ball?
 */
const movePiece = (piece: Piece, position: Position): boolean => {
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

  return isPickingUpBall;
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
 * @returns Did we pick up a loose ball?
 */
const activateGoalie = (goalie: Piece, position: Position): boolean => {
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

  return isPickingUpBall;
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

  endTurn();
};

const turnPiece = (piece: Piece, direction: FacingDirection): void => {
  piece.setFacingDirection(direction);

  useGameStore.setState({
    showDirectionArrows: false,
    selectedPiece: null,
    isSelectionLocked: false,
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
