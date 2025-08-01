"use client";

import { create } from "zustand";
import { Piece } from "@/classes/Piece";
import { Position } from "@/classes/Position";
import {
  FacingDirection,
  PlayerColor,
  SquareType,
  TutorialStep,
} from "@/types/types";
import {
  BOARD_COLS,
  BOARD_ROWS,
  DIRECTION_VECTORS,
  TUTORIAL_OPPONENT_COLOR,
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
  selectedPiece: Piece | null;
  /** Are we waiting for a user to select a direction for the piece **/
  awaitingDirectionSelection: boolean;
  /** Are we waiting for the user to consecutively pass */
  awaitingConsecutivePass: boolean;
  /* Is the turn button enabled */
  isTurnButtonEnabled: boolean;
  /* Whether to enable movement or not. This is used for tutorial steps */
  isMovementEnabled: boolean;
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
  "turning",
  "movement_with_ball",
  "passing",
  "consecutive_pass",
  "completed",
];

const demoPiece1 = new Piece(
  "W1",
  TUTORIAL_PLAYER_COLOR,
  new Position(4, 4),
  false,
);

/**
 * Predefined states for each tutorial step. This represents the state we should update. Anything not set won't get updated
 */
const tutorialStepStates: Record<TutorialStep, () => void> = {
  welcome: () => {
    useTutorialStore.setState({
      currentStep: "welcome",
      pieces: [],
      selectedPiece: null,
    });
  },
  basic_movement: () => {
    useTutorialStore.setState({
      currentStep: "basic_movement",
    });

    setBoardLayout([demoPiece1]);
  },
  turning: () => {
    demoPiece1.setHasBall(true);

    useTutorialStore.setState({
      currentStep: "turning",
      isMovementEnabled: false,
    });
  },
  movement_with_ball: () => {
    useTutorialStore.setState({
      currentStep: "movement_with_ball",
      isMovementEnabled: true,
    });
  },
  passing: () => {
    demoPiece1.setPosition(new Position(4, 4));
    demoPiece1.setFacingDirection("south");

    useTutorialStore.setState({
      currentStep: "passing",
      isMovementEnabled: false,
    });

    setBoardLayout([
      demoPiece1,
      new Piece("W2", TUTORIAL_PLAYER_COLOR, new Position(8, 0), false),
      new Piece("W3", TUTORIAL_PLAYER_COLOR, new Position(8, 4), false),
      new Piece("W4", TUTORIAL_PLAYER_COLOR, new Position(8, 8), false),
    ]);
  },
  consecutive_pass: () => {
    demoPiece1.setPosition(new Position(4, 4));
    demoPiece1.setFacingDirection("south");
    demoPiece1.setHasBall(true);

    useTutorialStore.setState({
      selectedPiece: null,
      currentStep: "consecutive_pass",
      isMovementEnabled: false,
    });

    setBoardLayout([
      demoPiece1,
      new Piece("W2", TUTORIAL_PLAYER_COLOR, new Position(8, 0), false),
      new Piece("B1", TUTORIAL_OPPONENT_COLOR, new Position(5, 4), false),
      new Piece("W3", TUTORIAL_PLAYER_COLOR, new Position(8, 4), false),
    ]);
  },
  completed: () => {
    useTutorialStore.setState({
      currentStep: "completed",
      selectedPiece: null,
      isMovementEnabled: false,
    });
  },
};

const useTutorialStore = create<TutorialState>(() => ({
  pieces: [],
  boardLayout: createBlankBoard(),
  awaitingDirectionSelection: false,
  awaitingConsecutivePass: false,
  selectedPiece: null,
  isTurnButtonEnabled: false,
  isMovementEnabled: true,
  currentStep: "welcome",
  completedSteps: new Set<TutorialStep>(),
  tutorialActive: false,
}));

/**
 * Gets all valid movement positions for a piece. Accounts for blocked paths due to other pieces
 * @returns Array of valid target positions
 */
const getValidMovementTargets = (piece: Piece): Position[] => {
  const { boardLayout } = useTutorialStore.getState();

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
 * Deselect the currently selected piece
 */
const deselectPiece = () => {
  const { selectedPiece } = useTutorialStore.getState();

  if (!selectedPiece) {
    // This should never happen. This is a bug if this gets called.
    throw new Error("Selected piece called when no piece was selected");
  }

  useTutorialStore.setState({
    awaitingDirectionSelection: false,
    selectedPiece: null,
    isTurnButtonEnabled: false,
  });
};

/**
 * Moves a piece from one position to another
 * @param piece - Piece to move
 * @param newPosition - Target position for the piece
 * @throws Error if piece exists at new position
 */
const movePiece = (piece: Piece, newPosition: Position) => {
  if (getPieceAtPosition(newPosition)) {
    // This should never happen, as we should check validation
    throw new Error("There can't be a piece at the new location.");
  }

  const [oRow, oCol] = piece.getPosition().getPositionCoordinates();
  const [nRow, nCol] = newPosition.getPositionCoordinates();

  piece.setPosition(newPosition);

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
 * Checks if a position is a valid movement target for the selected piece
 * @param position - The position to validate
 * @returns True if the position is a valid movement target
 */
export const isPositionValidMovementTarget = (position: Position): boolean => {
  const { selectedPiece } = useTutorialStore.getState();

  if (!selectedPiece) {
    return false;
  }

  const targets = getValidMovementTargets(selectedPiece);
  return !!targets.find((target) => target.equals(position));
};

/**
 * Get the info for a square
 * @param position Current square position
 * @param currentPlayerColor Current player color
 * @return The type that the square should display
 */
export const getSquareInfo = (
  position: Position,
  currentPlayerColor: PlayerColor,
): SquareType => {
  const state = useTutorialStore.getState();

  // Tutorial is complete, don't let anything happen
  if (state.currentStep === "completed") return "nothing";

  // If we are waiting direction selection, the only actions are to turn, transfer selection, or deselect
  if (state.awaitingDirectionSelection) {
    const turnTargets = getTurnTargetsForSelectedPiece();

    const isSquareTurnTarget = !!turnTargets.find((e) =>
      e.position.equals(position),
    );

    return isSquareTurnTarget ? "turn_target" : "nothing";
  }

  const [row, col] = position.getPositionCoordinates();
  const piece = useTutorialStore.getState().boardLayout[row][col];

  if (piece && piece.getColor() === currentPlayerColor) {
    if (
      state.selectedPiece &&
      state.selectedPiece.getHasBall() &&
      getValidPassTargets(state.selectedPiece).find((p) => p.equals(position))
    ) {
      return "pass_target";
    }

    return "piece";
  }

  if (state.isMovementEnabled && isPositionValidMovementTarget(position)) {
    return "movement";
  }

  return "nothing";
};

type getTurnTargetsReturnType = {
  position: Position;
  direction: FacingDirection;
}[];

/**
 * Get the valid turn targets for the currently selected piece
 * @param piece Piece to get turn targets of
 */
const getTurnTargets = (piece: Piece): getTurnTargetsReturnType => {
  const [row, col] = piece.getPosition().getPositionCoordinates();

  const targets: getTurnTargetsReturnType = [];

  if (row - 1 >= 0)
    targets.push({ position: new Position(row - 1, col), direction: "north" });
  if (row + 1 < BOARD_ROWS)
    targets.push({ position: new Position(row + 1, col), direction: "south" });
  if (col - 1 >= 0)
    targets.push({ position: new Position(row, col - 1), direction: "west" });
  if (col + 1 < BOARD_COLS)
    targets.push({ position: new Position(row, col + 1), direction: "east" });

  return targets;
};

/**
 * Get all valid pass targets for a piece
 * @param origin Piece to get pass targets of
 */
const getValidPassTargets = (origin: Piece): Position[] => {
  const validMoves: Position[] = [];

  const [curRow, curCol] = origin.getPosition().getPositionCoordinates();
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

      if (getPieceAtPosition(newPosition)) {
        validMoves.push(newPosition);

        // Break as we can't pass behind a piece
        break;
      }
    }
  }

  return validMoves;
};

/**
 * Pass the ball
 * @param origin Origin piece to pass ball from
 * @param destination Destination to pass ball to
 */
const passBall = (origin: Position, destination: Position) => {
  const originPiece = getPieceAtPosition(origin);
  const destinationPiece = getPieceAtPosition(destination);

  if (!originPiece || !destinationPiece) {
    throw new Error("There must be a piece at both the origin and destination");
  }

  originPiece.setHasBall(false);
  destinationPiece.setHasBall(true);

  // Push an update to the state
  useTutorialStore.setState({});
};

/**
 * Get the turn targets for the currently selected piece
 */
const getTurnTargetsForSelectedPiece = () => {
  const { selectedPiece } = useTutorialStore.getState();

  if (!selectedPiece) {
    throw new Error("A piece must be selected.");
  }

  return getTurnTargets(selectedPiece);
};

/**
 * Handles clicking on a square in the tutorial board
 * @param position - The position that was clicked
 */
export const handleSquareClick = (position: Position): void => {
  const {
    awaitingDirectionSelection,
    selectedPiece,
    currentStep,
    awaitingConsecutivePass,
    isMovementEnabled,
  } = useTutorialStore.getState();
  const pieceAtPosition = getPieceAtPosition(position);

  // Tutorial is complete, don't let anything happen
  if (currentStep === "completed") return;

  // First thing to check is if we are awaiting a direction selection. If we are, the only action the user can take is to rotate the piece
  if (awaitingDirectionSelection) {
    if (!selectedPiece) {
      throw new Error(
        "Awaiting direction selection, but there is no selected piece. This should never happen",
      );
    }

    const turnTarget = getTurnTargetsForSelectedPiece().find((e) =>
      e.position.equals(position),
    );

    if (!turnTarget) {
      // Since the clicked position is not a valid turn target, we are trying to de select
      deselectPiece();
      return;
    }

    // Valid target, so turn piece
    selectedPiece.setFacingDirection(turnTarget.direction);

    // Turn is over
    useTutorialStore.setState({
      awaitingDirectionSelection: false,
      selectedPiece: null,
      isTurnButtonEnabled: false,
    });

    if (useTutorialStore.getState().currentStep === "turning") {
      nextStep();
    }

    return;
  }

  // If we are awaiting a consecutive pass, the only thing the user can do is make that consecutive pass
  if (awaitingConsecutivePass) {
    if (!selectedPiece) {
      throw new Error(
        "No selected piece, but we are awaiting a consecutive pass",
      );
    }

    if (
      !pieceAtPosition ||
      pieceAtPosition.getColor() !== TUTORIAL_PLAYER_COLOR
    )
      return;

    const passTargets = getValidPassTargets(selectedPiece);

    // The clicked square must be a pass target
    if (!passTargets.find((p) => p.equals(pieceAtPosition.getPosition())))
      return;

    // User is trying to pass
    passBall(selectedPiece.getPosition(), position);

    // If we just made our consecutive pass, move to next step
    if (currentStep === "consecutive_pass" && awaitingConsecutivePass) {
      nextStep();
      return;
    }

    // No other actions besides passing, so don't allow anything else
    return;
  }

  if (pieceAtPosition && pieceAtPosition.getColor() === TUTORIAL_PLAYER_COLOR) {
    if (selectedPiece) {
      // If we currently have a piece selected, and we clicked a piece that is our color, we might be trying to pass, so check
      const passTargets = getValidPassTargets(selectedPiece);

      if (passTargets.find((p) => p.equals(pieceAtPosition.getPosition()))) {
        // This is a valid pass

        passBall(selectedPiece.getPosition(), position);

        if (currentStep === "passing") {
          nextStep();
          return;
        }
      }

      useTutorialStore.setState({
        awaitingConsecutivePass: true,
        selectedPiece: pieceAtPosition,
      });

      return;
    }

    // If there wasn't a selected piece, or there was, but it wasn't a valid pass target, our default action is to transfer selection state over
    useTutorialStore.setState({ selectedPiece: pieceAtPosition });

    // Enable the turn button after we select the piece
    if (currentStep === "turning") {
      useTutorialStore.setState({ isTurnButtonEnabled: true });
    }

    return;
  }

  // No piece, must mean user is trying to move
  if (
    isMovementEnabled &&
    selectedPiece &&
    isPositionValidMovementTarget(position)
  ) {
    // We are trying to move
    movePiece(selectedPiece, position);
    deselectPiece();

    // If we completed the basic movement successfully, move to next step
    if (useTutorialStore.getState().currentStep === "basic_movement") {
      nextStep();
    } else if (
      useTutorialStore.getState().currentStep === "movement_with_ball"
    ) {
      nextStep();
    }

    return;
  }

  // We are trying to de select
  if (selectedPiece) {
    deselectPiece();
  }
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

  tutorialStepStates[nextStep]();

  useTutorialStore.setState({
    completedSteps: newCompletedSteps,
    selectedPiece: null,
    awaitingConsecutivePass: false,
  });
};

/**
 * Handle the turn piece button click
 */
export const handleTurnPiece = () => {
  const { selectedPiece, currentStep } = useTutorialStore.getState();

  if (!selectedPiece) return;

  if (currentStep !== "turning") return;

  useTutorialStore.setState({
    awaitingDirectionSelection: true,
  });
};

/**
 * Hook to access the tutorial store for reactive updates
 */
export const useTutorialBoard = () => {
  return useTutorialStore();
};
