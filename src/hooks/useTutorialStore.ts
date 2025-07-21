"use client";

import { create } from "zustand";
import { Piece } from "@/classes/Piece";
import { Position } from "@/classes/Position";
import { PlayerColor, TutorialStep } from "@/types/types";
import {
  BOARD_COLS,
  BOARD_ROWS,
  TUTORIAL_PLAYER_COLOR,
} from "@/utils/constants";

/**
 * State interface for the tutorial store
 */
interface TutorialState {
  /** Array of all pieces currently on the board */
  pieces: Piece[];
  /** 2D array representing the board layout with pieces or null for empty squares */
  boardLayout: (Piece | null)[][];
  /** Position of the currently selected piece, if any */
  selectedPiecePosition: Position | null;
  /** Current step in the tutorial progression */
  currentStep: TutorialStep;
  /** Set of tutorial steps that have been completed */
  completedSteps: Set<TutorialStep>;
}

/**
 * Creates a BOARD_ROWS x BOARD_COLS blank board filled with null values
 * @returns A 2D array representing an empty game board
 */
const createBlankBoard = () =>
  Array.from({ length: BOARD_ROWS }, () =>
    (Array(BOARD_COLS) as (Piece | null)[]).fill(null),
  );

/**
 * Array of tutorial steps in order
 */
export const stepOrder: TutorialStep[] = [
  "welcome",
  "basic_movement",
  "movement_with_ball",
  "completed",
];

/**
 * Predefined states for each tutorial step. This represents the state we should update. Anything not set won't get updated
 */
const tutorialStepStates: Record<TutorialStep, Partial<TutorialState>> = {
  welcome: {
    currentStep: "welcome",
    pieces: [],
    selectedPiecePosition: null,
  },
  basic_movement: {
    currentStep: "basic_movement",
    pieces: [new Piece("W1", TUTORIAL_PLAYER_COLOR, new Position(4, 4), false)],
    selectedPiecePosition: null,
  },
  movement_with_ball: {
    currentStep: "movement_with_ball",
    pieces: [new Piece("W1", TUTORIAL_PLAYER_COLOR, new Position(4, 4), true)],
    selectedPiecePosition: new Position(4, 4),
  },
  completed: {
    currentStep: "completed",
    pieces: [],
    selectedPiecePosition: null,
  },
};

const useTutorialStore = create<TutorialState>(() => ({
  pieces: [],
  boardLayout: createBlankBoard(),
  selectedPiecePosition: null,
  currentStep: "welcome",
  completedSteps: new Set<TutorialStep>(),
  tutorialActive: false,
}));

/**
 * Gets all valid movement positions for the currently selected piece
 * @returns Array of valid target positions
 */
const getValidMovementTargetsForSelectedPiece = (): Position[] => {
  const state = useTutorialStore.getState();
  const { selectedPiecePosition, boardLayout } = state;

  if (!selectedPiecePosition) return [];

  const piece = getPieceAtPosition(selectedPiecePosition);
  if (!piece) return [];

  const validMoves: Position[] = [];
  const allMoves = piece.getMovementTargets();

  for (const pos of allMoves) {
    const [pRow, pCol] = pos.getPositionCoordinates();
    if (boardLayout[pRow][pCol] === null) {
      validMoves.push(pos);
    }
  }

  return validMoves;
};

/**
 * Moves a piece from one position to another
 * @param oldPosition - Current position of the piece
 * @param newPosition - Target position for the piece
 * @throws Error if no piece exists at old position or piece exists at new position
 */
const movePiece = (oldPosition: Position, newPosition: Position) => {
  const piece = getPieceAtPosition(oldPosition);

  if (!piece) {
    throw new Error("There has to be a piece at the old position");
  }

  if (getPieceAtPosition(newPosition)) {
    throw new Error("There can't be a piece at the new location");
  }

  piece.setPosition(newPosition);

  const [oRow, oCol] = oldPosition.getPositionCoordinates();
  const [nRow, nCol] = newPosition.getPositionCoordinates();

  useTutorialStore.setState((state) => {
    const newBoardLayout = state.boardLayout.map((row) => [...row]);
    newBoardLayout[oRow][oCol] = null;
    newBoardLayout[nRow][nCol] = piece;

    return {
      boardLayout: newBoardLayout,
    };
  });
};

/**
 * Sets the board layout with the given pieces
 * @param pieces - Array of pieces to place on the board
 */
export const setBoardLayout = (pieces: Piece[]) => {
  const boardLayout = createBlankBoard();

  pieces.forEach((piece) => {
    const [row, col] = piece.getPosition().getPositionCoordinates();
    boardLayout[row][col] = piece;
  });

  useTutorialStore.setState({
    pieces: [...pieces],
    boardLayout,
    selectedPiecePosition: null,
  });
};

/**
 * Gets the piece at a specific position on the board
 * @param position - The position to check
 * @returns The piece at the position or null if empty
 */
export const getPieceAtPosition = (position: Position): Piece | null => {
  const state = useTutorialStore.getState();
  const [row, col] = position.getPositionCoordinates();
  return state.boardLayout[row][col];
};

/**
 * Gets the currently selected piece position
 * @returns The selected position or null if no piece is selected
 */
export const getSelectedPosition = (): Position | null => {
  return useTutorialStore.getState().selectedPiecePosition;
};

/**
 * Checks if a position is a valid movement target for the selected piece
 * @param position - The position to validate
 * @returns True if the position is a valid movement target
 */
export const isPositionValidMovementTarget = (position: Position): boolean => {
  const state = useTutorialStore.getState();
  if (!state.selectedPiecePosition) return false;

  const targets = getValidMovementTargetsForSelectedPiece();
  return !!targets.find((target) => target.equals(position));
};

/**
 * Determines if a square can be clicked by the current player
 * @param position - The position to check
 * @param currentPlayerColor - The color of the current player
 * @returns True if the square is clickable
 */
export const isSquareClickable = (
  position: Position,
  currentPlayerColor: PlayerColor,
): boolean => {
  const [row, col] = position.getPositionCoordinates();
  const piece = useTutorialStore.getState().boardLayout[row][col];

  if (piece && piece.getColor() === currentPlayerColor) {
    return true;
  }

  return isPositionValidMovementTarget(position);
};

/**
 * Handles clicking on a square in the tutorial board
 * @param position - The position that was clicked
 */
export const handleSquareClick = (position: Position): void => {
  const state = useTutorialStore.getState();
  const piece = getPieceAtPosition(position);

  if (piece) {
    // Transfer selection over
    useTutorialStore.setState({ selectedPiecePosition: position });
    return;
  }

  // No piece, must mean user is trying to move or de select
  if (state.selectedPiecePosition && isPositionValidMovementTarget(position)) {
    // We are trying to move
    movePiece(state.selectedPiecePosition, position);
    useTutorialStore.setState({ selectedPiecePosition: null });

    // If we completed the basic movement successfully, move to next step
    if (useTutorialStore.getState().currentStep === "basic_movement") {
      nextStep();
    } else if (useTutorialStore.getState().currentStep === "movement_with_ball") {
      nextStep();
    }

    return;
  }

  // We are trying to de select
  useTutorialStore.setState({ selectedPiecePosition: null });
};

/**
 * Resets the board to an empty state
 */
export const resetBoard = () => {
  setBoardLayout([]);
};

/**
 * Advances to the next step in the tutorial
 * @throws Error if already at the last step
 */
export const nextStep = () => {
  const state = useTutorialStore.getState();
  const currentIndex = stepOrder.indexOf(state.currentStep);

  if (currentIndex >= stepOrder.length - 1) {
    throw new Error("There are no more steps left to complete.");
  }

  const nextStep = stepOrder[currentIndex + 1];
  const newCompletedSteps = new Set(state.completedSteps);
  newCompletedSteps.add(state.currentStep);

  const stepState = tutorialStepStates[nextStep];

  useTutorialStore.setState({
    completedSteps: newCompletedSteps,
    ...stepState,
  });
  if (stepState.pieces) setBoardLayout(stepState.pieces);
};

/**
 * Resets the tutorial to the welcome step
 */
export const resetTutorial = () => {
  const state = tutorialStepStates["welcome"];

  useTutorialStore.setState(state);
  if (state.pieces) setBoardLayout(state.pieces);
};

/**
 * Custom hook that provides access to tutorial board state and actions
 * @returns Object containing board state, tutorial state, and action functions
 */
export function useTutorialBoard() {
  const {
    pieces,
    boardLayout,
    selectedPiecePosition,
    currentStep,
    completedSteps,
  } = useTutorialStore();

  return {
    // Board state
    pieces,
    boardLayout,
    selectedPiecePosition,

    // Tutorial state
    currentStep,
    completedSteps,

    // Board actions
    getPieceAtPosition,
    getSelectedPosition,
    isPositionValidMovementTarget,
    isSquareClickable,
    handleSquareClick,
    resetBoard,

    // Tutorial actions
    nextStep,
    resetTutorial,
  };
}
