"use client";

import { create } from "zustand";
import { Piece } from "@/classes/Piece";
import { Position } from "@/classes/Position";
import {
  BoardSquareType,
  BoardType,
  FacingDirection,
  PlayerColor,
  SquareType,
  TutorialStep,
} from "@/types/types";
import {
  TUTORIAL_OPPONENT_COLOR,
  TUTORIAL_PLAYER_COLOR,
} from "@/utils/constants";
import {
  getTurnTargets,
  getValidEmptySquarePassTargets,
  getValidPassTargets,
  getValidTackleTargets,
  isPositionValidMovementTarget as isValidMovementTarget,
  isPlayerOffside,
} from "@/services/gameValidation";
import {
  createBlankBoard,
  createBoardLayout,
  findBall,
  findBallPosition,
  getAdjacentPieces,
  getAdjacentPositions,
  getBoardSquare as getBoardSquareHelper,
  getPieceAtPosition as getPieceAtPositionHelper,
  movePieceOnBoard,
  placeBallAtPosition as placeBallAtPositionHelper,
  swapPiecePositions,
} from "@/services/boardHelpers";

/**
 * State interface for the tutorial store
 */
interface TutorialState {
  /** Array of all pieces currently on the board */
  pieces: Piece[];
  /** 2D array representing the board layout with pieces or null for empty squares */
  boardLayout: BoardType;
  /** Position of the currently selected piece, if any */
  selectedPiece: Piece | null;
  /** Are we waiting for a user to select a direction for the piece **/
  awaitingDirectionSelection: boolean;
  /** Are we waiting for the user to consecutively pass */
  awaitingConsecutivePass: boolean;
  /** The person that just sent a pass that needs to be picked up **/
  passSender: Piece | null;
  /** Are we waiting for the user to receive a pass from an empty square */
  awaitingReceivePass: boolean;
  /** Show retry button for invalid pass attempts */
  showRetryButton?: boolean;
  /* Is the turn button enabled */
  isTurnButtonEnabled: boolean;
  /* Whether to enable movement or not. This is used for tutorial steps */
  isMovementEnabled: boolean;
  /** Current step in the tutorial progression */
  currentStep: TutorialStep;
  /** Set of tutorial steps that have been completed */
  completedSteps: Set<TutorialStep>;
  /** The unactivated white goalie piece to show at intersection */
  whiteUnactivatedGoaliePiece: Piece | null;
  /** Drag state for ball movement */
  isDragging: boolean;
  draggedPiece: Piece | null;
  dragStartPosition: Position | null;
}

/**
 * Array of tutorial steps in order
 */
export const stepOrder: TutorialStep[] = [
  "welcome",
  "basic_movement",
  "movement_with_ball",
  "turning",
  "passing",
  "ball_empty_square",
  "ball_pickup",
  "receiving_passes",
  "chip_pass",
  "consecutive_pass",
  "shooting",
  "consecutive_pass_to_score",
  "tackling",
  "tackling_positioning",
  "activating_goalies",
  "blocking_shots",
  "offside",
  "shooting_zone_pass",
  "completed",
];

const WHITE_GOALIE_ACTIVATION_TARGETS: Position[] = [
  new Position(0, 3),
  new Position(0, 4),
  new Position(0, 5),
  new Position(0, 6),
  new Position(1, 3),
  new Position(1, 4),
  new Position(1, 5),
  new Position(1, 6),
  new Position(2, 3),
  new Position(3, 2),
  new Position(2, 6),
  new Position(3, 7),
  new Position(2, 4),
  new Position(3, 4),
  new Position(2, 5),
  new Position(3, 5),
];

const demoPiece1 = new Piece({
  id: "W1",
  color: TUTORIAL_PLAYER_COLOR,
  position: new Position(4, 4),
  hasBall: false,
});

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
  movement_with_ball: () => {
    demoPiece1.setHasBall(true);

    useTutorialStore.setState({
      currentStep: "movement_with_ball",
      isMovementEnabled: true,
    });
  },
  turning: () => {
    useTutorialStore.setState({
      currentStep: "turning",
      isMovementEnabled: false,
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
      new Piece({
        id: "W2",
        color: TUTORIAL_PLAYER_COLOR,
        position: new Position(8, 0),
        hasBall: false,
      }),
      new Piece({
        id: "W3",
        color: TUTORIAL_PLAYER_COLOR,
        position: new Position(8, 4),
        hasBall: false,
      }),
      new Piece({
        id: "W4",
        color: TUTORIAL_PLAYER_COLOR,
        position: new Position(8, 8),
        hasBall: false,
      }),
      new Piece({
        id: "W5",
        color: TUTORIAL_PLAYER_COLOR,
        position: new Position(4, 0),
        hasBall: false,
      }),
      new Piece({
        id: "W6",
        color: TUTORIAL_PLAYER_COLOR,
        position: new Position(4, 8),
        hasBall: false,
      }),
      new Piece({
        id: "W7",
        color: TUTORIAL_PLAYER_COLOR,
        position: new Position(1, 4),
        hasBall: false,
      }),
    ]);
  },
  consecutive_pass: () => {
    // Demo piece 1 will be the final piece to receive ball
    demoPiece1.setPosition(new Position(8, 4));
    demoPiece1.setFacingDirection("south");
    demoPiece1.setHasBall(false);

    useTutorialStore.setState({
      currentStep: "consecutive_pass",
      isMovementEnabled: false,
    });

    setBoardLayout([
      new Piece({
        id: "W3",
        color: TUTORIAL_PLAYER_COLOR,
        position: new Position(4, 4),
        hasBall: true,
      }),
      new Piece({
        id: "W2",
        color: TUTORIAL_PLAYER_COLOR,
        position: new Position(8, 0),
        hasBall: false,
      }),
      new Piece({
        id: "B1",
        color: TUTORIAL_OPPONENT_COLOR,
        position: new Position(5, 4),
        hasBall: false,
      }),
      demoPiece1,
    ]);
  },
  consecutive_pass_to_score: () => {
    // Set up a scenario where the user needs to make consecutive passes to score
    // Position: piece with ball at (10, 4), opponent blocking at (11, 4), teammate at (10, 6) to pass to horizontally
    useTutorialStore.setState({
      currentStep: "consecutive_pass_to_score",
      isMovementEnabled: false,
    });

    setBoardLayout([
      new Piece({
        id: "W1",
        color: TUTORIAL_PLAYER_COLOR,
        position: new Position(10, 4),
        hasBall: true,
        facingDirection: "east",
      }),
      new Piece({
        id: "W2",
        color: TUTORIAL_PLAYER_COLOR,
        position: new Position(10, 6),
        hasBall: false,
        facingDirection: "south",
      }),
      new Piece({
        id: "B1",
        color: TUTORIAL_OPPONENT_COLOR,
        position: new Position(11, 4),
        hasBall: false,
      }),
    ]);
  },
  ball_empty_square: () => {
    demoPiece1.setFacingDirection("south");
    demoPiece1.setHasBall(true);

    useTutorialStore.setState({
      currentStep: "ball_empty_square",
      isMovementEnabled: false,
    });

    setBoardLayout([demoPiece1]);
  },
  ball_pickup: () => {
    // Position the demo piece at (4,4) and place a ball at (6,4)
    demoPiece1.setPosition(new Position(4, 4));
    demoPiece1.setHasBall(false);

    useTutorialStore.setState({
      currentStep: "ball_pickup",
      isMovementEnabled: true,
      selectedPiece: null,
    });

    // Set up board layout with piece at (4,4) and ball at (6,4)
    setBoardLayout([demoPiece1], [new Position(6, 4)]);
  },
  receiving_passes: () => {
    // Set up a scenario where player passes to empty square and needs to receive
    // Place piece with ball at (4,4) and another piece at (7,4) to receive
    demoPiece1.setPosition(new Position(4, 4));
    demoPiece1.setHasBall(true);
    demoPiece1.setFacingDirection("south");

    useTutorialStore.setState({
      currentStep: "receiving_passes",
      isMovementEnabled: false,
    });

    // Place pieces so that there are no direct passing targets at (6,4)
    // This forces the user to pass to an empty square
    setBoardLayout([
      demoPiece1,
      new Piece({
        id: "W2",
        color: TUTORIAL_PLAYER_COLOR,
        position: new Position(7, 3),
        hasBall: false,
      }),
    ]);
  },
  chip_pass: () => {
    useTutorialStore.setState({
      currentStep: "chip_pass",
      isMovementEnabled: false,
    });

    setBoardLayout([
      new Piece({
        id: "W1",
        color: TUTORIAL_PLAYER_COLOR,
        position: new Position(4, 4),
        hasBall: true,
      }),
      new Piece({
        id: "W2",
        color: TUTORIAL_PLAYER_COLOR,
        position: new Position(8, 4),
        hasBall: false,
      }),
      new Piece({
        id: "W3",
        color: TUTORIAL_PLAYER_COLOR,
        position: new Position(4, 1),
        hasBall: false,
      }),
      new Piece({
        id: "W4",
        color: TUTORIAL_PLAYER_COLOR,
        position: new Position(4, 8),
        hasBall: false,
      }),
      new Piece({
        id: "B1",
        color: TUTORIAL_OPPONENT_COLOR,
        position: new Position(4, 3),
        hasBall: false,
      }),
      new Piece({
        id: "B2",
        color: TUTORIAL_OPPONENT_COLOR,
        position: new Position(4, 5),
        hasBall: false,
      }),
      new Piece({
        id: "B3",
        color: TUTORIAL_OPPONENT_COLOR,
        position: new Position(6, 4),
        hasBall: false,
      }),
    ]);
  },
  shooting: () => {
    useTutorialStore.setState({
      currentStep: "shooting",
      isMovementEnabled: false,
    });

    setBoardLayout([
      new Piece({
        id: "W1",
        color: TUTORIAL_PLAYER_COLOR,
        position: new Position(9, 4),
        hasBall: true,
      }),
    ]);
  },
  tackling: () => {
    useTutorialStore.setState({
      currentStep: "tackling",
      isMovementEnabled: false,
    });

    // Set up tackling scenario: white piece adjacent to black piece with ball
    // Black piece faces east (perpendicular to white piece) so tackle is allowed
    setBoardLayout([
      new Piece({
        id: "W1",
        color: TUTORIAL_PLAYER_COLOR,
        position: new Position(6, 4),
        hasBall: false,
      }),
      new Piece({
        id: "B1",
        color: TUTORIAL_OPPONENT_COLOR,
        position: new Position(6, 5),
        hasBall: true,
        facingDirection: "west",
      }),
      new Piece({
        id: "B2",
        color: TUTORIAL_OPPONENT_COLOR,
        position: new Position(6, 3),
        hasBall: true,
        facingDirection: "west",
      }),
    ]);
  },
  tackling_positioning: () => {
    useTutorialStore.setState({
      currentStep: "tackling_positioning",
      isMovementEnabled: true,
    });

    // Set up positioning scenario: player must maneuver around opponent to tackle
    // Black piece faces east (away from white piece) and is adjacent
    // White piece starts at (6, 4), black piece at (6, 5) facing east
    // Player needs to move to a valid tackle position (not from behind)
    setBoardLayout([
      new Piece({
        id: "W1",
        color: TUTORIAL_PLAYER_COLOR,
        position: new Position(6, 4),
        hasBall: false,
      }),
      new Piece({
        id: "B1",
        color: TUTORIAL_OPPONENT_COLOR,
        position: new Position(6, 5),
        hasBall: true,
        facingDirection: "east",
      }),
    ]);
  },
  activating_goalies: () => {
    const goalie = new Piece({
      id: "WG",
      color: TUTORIAL_PLAYER_COLOR,
      position: "white_unactivated",
      hasBall: false,
      isGoalie: true,
    });

    useTutorialStore.setState({
      currentStep: "activating_goalies",
      isMovementEnabled: false,
      whiteUnactivatedGoaliePiece: goalie,
    });

    // Set up goalie activation scenario - show unactivated goalie at intersection
    setBoardLayout([]);
  },
  blocking_shots: () => {
    const goalie = new Piece({
      id: "WG",
      color: TUTORIAL_PLAYER_COLOR,
      position: "white_unactivated",
      hasBall: false,
      isGoalie: true,
    });

    useTutorialStore.setState({
      currentStep: "blocking_shots",
      isMovementEnabled: false,
      whiteUnactivatedGoaliePiece: goalie,
    });

    // Set up blocking scenario: black piece with ball on (3,2), need to activate goalie on (1,4)
    setBoardLayout([
      new Piece({
        id: "B1",
        color: TUTORIAL_OPPONENT_COLOR,
        position: new Position(3, 2),
        hasBall: true,
      }),
    ]);
  },
  offside: () => {
    useTutorialStore.setState({
      currentStep: "offside",
      isMovementEnabled: false,
    });

    // Set up offside scenario:
    // - White piece with ball at (6,4) facing south
    // - White piece at (10,3) in offside position (will NOT be highlighted)
    // - White piece at (8,3) in onside position (WILL be highlighted)
    // - Black pieces positioned to create offside situation
    setBoardLayout([
      new Piece({
        id: "W1",
        color: TUTORIAL_PLAYER_COLOR,
        position: new Position(6, 4),
        hasBall: true,
        facingDirection: "south",
      }),
      new Piece({
        id: "W2",
        color: TUTORIAL_PLAYER_COLOR,
        position: new Position(10, 0), // Offside position
        hasBall: false,
      }),
      new Piece({
        id: "W3",
        color: TUTORIAL_PLAYER_COLOR,
        position: new Position(9, 7), // Onside position
        hasBall: false,
      }),
      new Piece({
        id: "B1",
        color: TUTORIAL_OPPONENT_COLOR,
        position: new Position(9, 4), // Closest to own goal
        hasBall: false,
      }),
      new Piece({
        id: "B2",
        color: TUTORIAL_OPPONENT_COLOR,
        position: new Position(9, 6), // Second-to-last opponent
        hasBall: false,
      }),
      new Piece({
        id: "B3",
        color: TUTORIAL_OPPONENT_COLOR,
        position: new Position(7, 4), // Third opponent
        hasBall: false,
      }),
    ]);
  },
  shooting_zone_pass: () => {
    useTutorialStore.setState({
      currentStep: "shooting_zone_pass",
      isMovementEnabled: false,
    });

    // Set up scenario demonstrating cross-zone pass rule
    // White's shooting zone: rows 9-13 (closest to black's goal at row 13)
    // Black's shooting zone: rows 0-4 (closest to white's goal at row 0)
    // Player is in white's shooting zone and needs to pass across to black's zone
    setBoardLayout([
      new Piece({
        id: "W1",
        color: TUTORIAL_PLAYER_COLOR,
        position: new Position(4, 4), // In white's shooting zone (rows 9-13)
        hasBall: true,
        facingDirection: "south",
      }),
      new Piece({
        id: "W2",
        color: TUTORIAL_PLAYER_COLOR,
        position: new Position(9, 4), // Teammate in black's zone (rows 0-4) - crosses zones
        hasBall: false,
      }),
      new Piece({
        id: "W3",
        color: TUTORIAL_PLAYER_COLOR,
        position: new Position(4, 9), // Teammate in same white zone - no zone crossing
        hasBall: false,
      }),
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
  awaitingReceivePass: false,
  passSender: null,
  selectedPiece: null,
  isTurnButtonEnabled: false,
  isMovementEnabled: true,
  currentStep: "welcome",
  completedSteps: new Set<TutorialStep>(),
  tutorialActive: false,
  showRetryButton: false,
  whiteUnactivatedGoaliePiece: null,
  isDragging: false,
  draggedPiece: null,
  dragStartPosition: null,
}));

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
  useTutorialStore.setState((state) => {
    const newBoardLayout = movePieceOnBoard(
      piece,
      newPosition,
      state.boardLayout,
    );
    return { boardLayout: newBoardLayout };
  });
};

/**
 * Pass the ball
 * @param origin Origin piece to pass ball from
 * @param destination Destination to pass ball to
 */
const passBall = (origin: Position, destination: Position) => {
  const originPiece = getPieceAtPosition(origin);
  const destinationPiece = getPieceAtPosition(destination);

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
  } else {
    placeBallAtPosition(destination);
  }

  // Push an update to the state
  useTutorialStore.setState({});
};

/**
 * Handle turn target clicks during direction selection
 */
const handleTurnTarget = (position: Position): void => {
  const { selectedPiece, currentStep } = useTutorialStore.getState();

  if (!selectedPiece) {
    throw new Error(
      "Awaiting direction selection, but there is no selected piece. This should never happen",
    );
  }

  const turnTarget = getTurnTargets(selectedPiece).find((e) =>
    e.position.equals(position),
  );

  if (!turnTarget) {
    // Since the clicked position is not a valid turn target, we are trying to deselect
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

  if (
    currentStep === "turning" ||
    currentStep === "receiving_passes" ||
    currentStep === "tackling" ||
    currentStep === "tackling_positioning" ||
    currentStep === "passing" ||
    currentStep === "consecutive_pass" ||
    currentStep === "ball_pickup" ||
    currentStep === "chip_pass" ||
    currentStep === "offside" ||
    currentStep === "shooting_zone_pass"
  ) {
    nextStep();
  }
};

/**
 * Checks if a pass crosses shooting zones (full move rule)
 */
const isCrossZonePass = (
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

/**
 * Handle piece selection and potential passing
 */
const handlePieceSelection = (position: Position): void => {
  const { selectedPiece, currentStep } = useTutorialStore.getState();
  const pieceAtPosition = getPieceAtPosition(position);

  if (
    !pieceAtPosition ||
    pieceAtPosition.getColor() !== TUTORIAL_PLAYER_COLOR
  ) {
    return;
  }

  // Check if this is a pass target
  if (selectedPiece && selectedPiece.getHasBall()) {
    const state = useTutorialStore.getState();
    const checkOffside = state.currentStep === "offside";
    const passTargets = getValidPassTargets(
      selectedPiece,
      state.boardLayout,
      checkOffside,
    );

    if (
      passTargets.some((p) =>
        p.equals(pieceAtPosition.getPositionOrThrowIfUnactivated()),
      )
    ) {
      // This is a valid pass
      const fromPosition = selectedPiece.getPositionOrThrowIfUnactivated();
      passBall(fromPosition, position);

      // Select the piece that received the pass
      useTutorialStore.setState({
        selectedPiece: pieceAtPosition,
      });

      // Check if this is a cross-zone pass for shooting_zone_pass step
      if (
        currentStep === "shooting_zone_pass" &&
        isCrossZonePass(fromPosition, position)
      ) {
        // Cross-zone pass - only allow direction selection, no consecutive passes
        useTutorialStore.setState({
          awaitingDirectionSelection: true,
        });
        return;
      }

      if (
        currentStep === "consecutive_pass" ||
        currentStep === "consecutive_pass_to_score" ||
        (currentStep === "shooting_zone_pass" &&
          !isCrossZonePass(fromPosition, position))
      ) {
        // Allow consecutive passes for same-zone passes in shooting_zone_pass step
        useTutorialStore.setState({
          awaitingConsecutivePass: true,
        });
      } else if (
        currentStep === "passing" ||
        currentStep === "chip_pass" ||
        currentStep === "offside" ||
        (currentStep === "shooting_zone_pass" &&
          isCrossZonePass(fromPosition, position))
      ) {
        useTutorialStore.setState({
          awaitingDirectionSelection: true,
        });
      }

      return;
    }
  }

  // Default action: select the piece
  useTutorialStore.setState({ selectedPiece: pieceAtPosition });

  // Enable the turn button after we select the piece
  if (currentStep === "turning") {
    useTutorialStore.setState({ isTurnButtonEnabled: true });
  }
};

/**
 * Handle pass target clicks during consecutive passing
 */
const handleConsecutivePass = (position: Position): void => {
  const { selectedPiece } = useTutorialStore.getState();

  if (!selectedPiece) {
    throw new Error(
      "No selected piece, but we are awaiting a consecutive pass",
    );
  }

  if (!selectedPiece.getHasBall()) {
    throw new Error(
      "There is a selected piece, but it doesn't have the ball and we are awaiting a consecutive pass",
    );
  }

  const pieceAtPosition = getPieceAtPosition(position);

  if (
    !pieceAtPosition ||
    pieceAtPosition.getColor() !== TUTORIAL_PLAYER_COLOR
  ) {
    return;
  }

  const state = useTutorialStore.getState();
  const checkOffside = state.currentStep === "offside";
  const passTargets = getValidPassTargets(
    selectedPiece,
    state.boardLayout,
    checkOffside,
  );

  // The clicked square must be a pass target
  if (
    !passTargets.some((p) =>
      p.equals(pieceAtPosition.getPositionOrThrowIfUnactivated()),
    )
  ) {
    return;
  }

  // User is trying to pass
  passBall(selectedPiece.getPositionOrThrowIfUnactivated(), position);

  // If we just made our consecutive pass, move to next step
  useTutorialStore.setState({
    awaitingDirectionSelection: true,
    selectedPiece: pieceAtPosition,
  });
};

/**
 * Handle movement to empty squares or squares with balls
 */
const handleMovement = (position: Position): void => {
  const { selectedPiece, currentStep, awaitingReceivePass, isDragging } =
    useTutorialStore.getState();

  if (!selectedPiece) {
    return;
  }

  const state = useTutorialStore.getState();

  // Prevent click-based movement for pieces with ball (they should use drag)
  // Exception: allow if this is called from handleBallDrop (isDragging will be true)
  if (selectedPiece.getHasBall() && !isDragging) {
    return;
  }

  // Check if we're moving the unactivated goalie to the board
  if (selectedPiece === state.whiteUnactivatedGoaliePiece) {
    // Update the piece's position
    selectedPiece.setPosition(position);

    // Add the piece to the board layout and pieces array
    const newPieces = [...state.pieces, selectedPiece];
    setBoardLayout(newPieces);

    // Clear the unactivated goalie state and deselect
    useTutorialStore.setState({
      whiteUnactivatedGoaliePiece: null,
      selectedPiece: null,
    });

    // Progress to next step after goalie activation
    if (currentStep === "activating_goalies") {
      nextStep();
    } else if (currentStep === "blocking_shots") {
      // Check if goalie was placed on a valid position to block the shot
      const validPositions = [new Position(1, 4), new Position(0, 5)];
      if (validPositions.some((pos) => position.equals(pos))) {
        nextStep();
      } else {
        // Wrong position - show retry button and reset the goalie
        const goalie = new Piece({
          id: "WG",
          color: TUTORIAL_PLAYER_COLOR,
          position: "white_unactivated",
          hasBall: false,
          isGoalie: true,
        });

        useTutorialStore.setState({
          whiteUnactivatedGoaliePiece: goalie,
          selectedPiece: null,
          showRetryButton: true,
        });

        // Remove the incorrectly placed goalie from the board
        setBoardLayout([
          new Piece({
            id: "B1",
            color: TUTORIAL_OPPONENT_COLOR,
            position: new Position(3, 2),
            hasBall: true,
          }),
        ]);
      }
    }
    return;
  }

  const boardSquare = getBoardSquareHelper(position, state.boardLayout);
  const isPickingUpBall = boardSquare === "ball";

  movePiece(selectedPiece, position);

  // Handle receiving pass completion
  if (awaitingReceivePass && isPickingUpBall) {
    useTutorialStore.setState({
      awaitingDirectionSelection: true,
    });
    return;
  }

  if (isPickingUpBall) {
    useTutorialStore.setState({
      awaitingDirectionSelection: true,
    });
  } else {
    deselectPiece();
  }

  // Check for step progression
  if (
    currentStep === "basic_movement" ||
    currentStep === "movement_with_ball"
  ) {
    nextStep();
  }
};

/**
 * Handle passing to empty squares
 */
const handleEmptySquarePass = (position: Position): void => {
  const { selectedPiece, currentStep, boardLayout } =
    useTutorialStore.getState();

  if (!selectedPiece) {
    return;
  }

  passBall(selectedPiece.getPositionOrThrowIfUnactivated(), position);

  if (currentStep === "ball_empty_square") {
    nextStep();
  } else if (currentStep === "receiving_passes") {
    const adjPieces = getAdjacentPieces(
      position,
      TUTORIAL_PLAYER_COLOR,
      boardLayout,
    ).filter((e) => e !== selectedPiece);

    // If there are no pieces nearby, show the retry button
    if (adjPieces.length === 0) {
      useTutorialStore.setState({
        selectedPiece: null,
        showRetryButton: true,
      });
      return;
    }

    // Valid pass - pieces are nearby to receive
    useTutorialStore.setState({
      awaitingReceivePass: true,
      selectedPiece: null,
      passSender: selectedPiece,
      isMovementEnabled: true,
    });
  } else if (
    currentStep === "shooting" ||
    currentStep === "consecutive_pass_to_score"
  ) {
    if (!position.isPositionInGoal()) {
      useTutorialStore.setState({
        selectedPiece: null,
        showRetryButton: true,
      });
      return;
    }

    nextStep();
  }
};

/**
 * Handle tackle target clicks
 */
const handleTackle = (position: Position): void => {
  const { selectedPiece } = useTutorialStore.getState();

  if (!selectedPiece) {
    return;
  }

  const targetPiece = getPieceAtPosition(position);

  if (!targetPiece || !targetPiece.getHasBall()) {
    return;
  }

  // Perform the tackle
  useTutorialStore.setState((state) => {
    const newBoardLayout = swapPiecePositions(
      selectedPiece,
      targetPiece,
      state.boardLayout,
    );
    return {
      boardLayout: newBoardLayout,
      awaitingDirectionSelection: true,
    };
  });
};

/**
 * Handle deselection when clicking on nothing
 */
const handleDeselection = (): void => {
  const { selectedPiece, currentStep, awaitingDirectionSelection } =
    useTutorialStore.getState();

  if (
    awaitingDirectionSelection &&
    (currentStep === "tackling" ||
      currentStep === "tackling_positioning" ||
      currentStep === "receiving_passes" ||
      currentStep === "passing" ||
      currentStep === "consecutive_pass" ||
      currentStep === "ball_pickup" ||
      currentStep === "chip_pass" ||
      currentStep === "offside")
  ) {
    return;
  }

  if (selectedPiece) {
    deselectPiece();
  }
};

/**
 * Reset the tutorial state. This should be called after every step transition.
 * @param completedSteps Optional param to update the completed steps
 */
const resetState = (completedSteps?: Set<TutorialStep>) => {
  const newState: Record<string, unknown> = {
    selectedPiece: null,
    awaitingConsecutivePass: false,
    awaitingReceivePass: false,
    showRetryButton: false,
    isDragging: false,
    draggedPiece: null,
    dragStartPosition: null,
  };

  if (completedSteps) newState.completedSteps = completedSteps;

  useTutorialStore.setState(newState);
};

/**
 * Sets the board layout with the given pieces
 * @param pieces - Array of pieces to place on the board
 * @param balls - Where to place balls
 */
export const setBoardLayout = (pieces: Piece[], balls?: Position[]) => {
  const boardLayout = createBoardLayout(pieces, balls);

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
  const { boardLayout } = useTutorialStore.getState();
  return getPieceAtPositionHelper(position, boardLayout);
};

/**
 * Place a ball at a given position
 * @param position Position to place ball
 */
const placeBallAtPosition = (position: Position): void => {
  useTutorialStore.setState((state) => {
    const newBoardLayout = placeBallAtPositionHelper(
      position,
      state.boardLayout,
    );
    return { boardLayout: newBoardLayout };
  });
};

/**
 * Checks if a position is a valid movement target for the selected piece
 * @param position - The position to validate
 * @returns True if the position is a valid movement target
 */
export const isPositionValidMovementTarget = (position: Position): boolean => {
  const { selectedPiece, boardLayout } = useTutorialStore.getState();

  if (!selectedPiece) {
    return false;
  }

  return isValidMovementTarget(selectedPiece, position, boardLayout);
};

/**
 * Checks if a piece is in an offside position
 * @param piece - The piece to check
 * @returns True if the piece is offside
 */
export const isPieceOffside = (piece: BoardSquareType): boolean => {
  // Safety check: only check offside for actual Piece instances
  if (!(piece instanceof Piece)) {
    return false;
  }

  const { boardLayout, currentStep } = useTutorialStore.getState();

  // Only show offside during the offside tutorial step
  if (currentStep !== "offside") {
    return false;
  }

  const ballPosition = findBallPosition(boardLayout);
  if (!ballPosition) {
    return false;
  }

  return isPlayerOffside(piece, ballPosition, boardLayout);
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

  if (state.showRetryButton) return "nothing";

  // Tutorial is complete, don't let anything happen
  if (state.currentStep === "completed") return "nothing";

  // If we are waiting direction selection, the only actions are to turn, transfer selection, or deselect
  if (state.awaitingDirectionSelection) {
    const turnTargets = state.selectedPiece
      ? getTurnTargets(state.selectedPiece)
      : [];

    const isSquareTurnTarget = turnTargets.some((e) =>
      e.position.equals(position),
    );

    return isSquareTurnTarget ? "turn_target" : "nothing";
  }

  // If unactivated white goalie is selected, show movement targets in goal area
  if (
    state.selectedPiece &&
    state.selectedPiece === state.whiteUnactivatedGoaliePiece
  ) {
    const isGoalActivationTarget = WHITE_GOALIE_ACTIVATION_TARGETS.some((e) =>
      e.equals(position),
    );

    // Only allow activation if the position is a valid target AND there's no piece there
    if (isGoalActivationTarget && !getPieceAtPosition(position)) {
      return "movement";
    }

    return "nothing";
  }

  const pieceAtPosition = getPieceAtPosition(position);

  // If we are awaiting receive pass, only allow clicking pieces within one square of ball
  if (state.awaitingReceivePass) {
    if (!state.selectedPiece) {
      const piece = getPieceAtPosition(position);

      if (!piece || piece.getColor() !== currentPlayerColor) return "nothing";

      // Find the ball
      const ballPos = findBall(state.boardLayout);

      if (!ballPos) return "nothing";

      const adjPiecesToBall = getAdjacentPieces(
        ballPos,
        TUTORIAL_PLAYER_COLOR,
        state.boardLayout,
      );

      if (adjPiecesToBall.some((p) => p === pieceAtPosition)) return "piece";

      return "nothing";
    } else {
      // Don't try to get adjacent positions for unactivated goalies
      if (state.selectedPiece === state.whiteUnactivatedGoaliePiece) {
        return "nothing";
      }

      const adjPositions = getAdjacentPositions(
        state.selectedPiece.getPositionOrThrowIfUnactivated(),
      );
      const ballPos = findBall(state.boardLayout);

      if (!ballPos) {
        throw new Error("Can't find ball on board");
      }

      const positionIsAdjToSelectedPiece =
        adjPositions.filter((p) => p.equals(ballPos) && p.equals(position))
          .length > 0;

      return positionIsAdjToSelectedPiece ? "movement" : "nothing";
    }
  }

  if (pieceAtPosition) {
    if (state.selectedPiece && state.selectedPiece.getHasBall()) {
      const checkOffside = state.currentStep === "offside";
      const passTargets = getValidPassTargets(
        state.selectedPiece,
        state.boardLayout,
        checkOffside,
      );

      const positionIsPassTarget = passTargets.find((p) => p.equals(position));

      if (positionIsPassTarget) return "pass_target";
    }

    // Check for tackle targets (only in tackling step)
    if (
      (state.currentStep === "tackling" ||
        state.currentStep === "tackling_positioning") &&
      state.selectedPiece &&
      !state.selectedPiece.getHasBall() &&
      pieceAtPosition.getColor() !== TUTORIAL_PLAYER_COLOR
    ) {
      const tackleTargets = getValidTackleTargets(
        state.selectedPiece,
        state.boardLayout,
      );

      const positionIsTackleTarget = tackleTargets.find((p) =>
        p.equals(position),
      );

      if (positionIsTackleTarget) return "tackle_target";
    }

    return pieceAtPosition.getColor() === TUTORIAL_PLAYER_COLOR
      ? "piece"
      : "nothing";
  }

  if (state.isMovementEnabled && isPositionValidMovementTarget(position)) {
    return "movement";
  }

  // Check for empty square pass targets (only in ball_empty_square and receiving_passes steps)
  if (
    (state.currentStep === "ball_empty_square" ||
      state.currentStep === "receiving_passes" ||
      state.currentStep === "shooting" ||
      state.currentStep === "consecutive_pass_to_score") &&
    state.selectedPiece &&
    state.selectedPiece.getHasBall() &&
    getValidEmptySquarePassTargets(state.selectedPiece, state.boardLayout).find(
      (p) => p.equals(position),
    )
  ) {
    return "empty_pass_target";
  }

  return "nothing";
};

/**
 * Handles clicking on a square in the tutorial board
 * @param position - The position that was clicked
 */
export const handleSquareClick = (position: Position): void => {
  const { currentStep, awaitingConsecutivePass, showRetryButton } =
    useTutorialStore.getState();

  // If we need to retry a step, don't let anything happen
  if (showRetryButton) return;

  // Tutorial is complete, don't let anything happen
  if (currentStep === "completed") return;

  // Use getSquareInfo to determine what type of square was clicked
  const squareType = getSquareInfo(position, TUTORIAL_PLAYER_COLOR);

  // Route to appropriate handler based on square type
  switch (squareType) {
    case "turn_target":
      return handleTurnTarget(position);
    case "pass_target":
      if (awaitingConsecutivePass) return handleConsecutivePass(position);

      return handlePieceSelection(position);
    case "piece":
      return handlePieceSelection(position);
    case "movement":
      return handleMovement(position);
    case "empty_pass_target":
      return handleEmptySquarePass(position);
    case "tackle_target":
      return handleTackle(position);
    case "nothing":
    default:
      return handleDeselection();
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
  resetState(newCompletedSteps);
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
 * Handle retry button click to reset the receiving passes step
 */
export const handleRetry = () => {
  const { currentStep } = useTutorialStore.getState();

  tutorialStepStates[currentStep]();
  resetState();
};

/**
 * Handle clicking on the unactivated goalie
 */
export const handleUnactivatedGoalieClick = (goaliePiece: Piece) => {
  const { selectedPiece } = useTutorialStore.getState();

  // If already selected, deselect
  if (selectedPiece === goaliePiece) {
    useTutorialStore.setState({ selectedPiece: null });
  } else {
    // Select the unactivated goalie
    useTutorialStore.setState({ selectedPiece: goaliePiece });
  }
};

/**
 * Handle start of ball drag
 */
export const handleBallDragStart = (
  piece: Piece,
  initialX?: number,
  initialY?: number,
) => {
  if (!piece.getHasBall()) return;

  useTutorialStore.setState({
    isDragging: true,
    draggedPiece: piece,
    dragStartPosition: piece.getPositionOrThrowIfUnactivated(),
    selectedPiece: piece,
  });

  // If initial position provided, dispatch a custom event to set initial mouse position
  if (initialX !== undefined && initialY !== undefined) {
    window.dispatchEvent(
      new CustomEvent("dragstart-position", {
        detail: { x: initialX, y: initialY },
      }),
    );
  }
};

/**
 * Handle ball drag over a position
 */
export const handleBallDragOver = (position: Position, event: DragEvent) => {
  const { draggedPiece } = useTutorialStore.getState();

  if (!draggedPiece || !isPositionValidMovementTarget(position)) {
    event.dataTransfer!.dropEffect = "none";
    return;
  }

  event.preventDefault();
  event.dataTransfer!.dropEffect = "move";
};

/**
 * Handle ball drop on a position
 */
export const handleBallDrop = (position: Position, event: DragEvent) => {
  event.preventDefault();

  const { draggedPiece } = useTutorialStore.getState();

  if (!draggedPiece) {
    handleBallDragEnd();
    return;
  }

  if (isPositionValidMovementTarget(position)) {
    // Valid drop - execute the movement
    handleMovement(position);
  } else {
    // Invalid drop - snap back to original position
    handleBallDragEnd();
  }
};

/**
 * Handle mouse-based ball drop
 */
export const handleMouseBallDrop = (position: Position) => {
  const { draggedPiece, dragStartPosition } = useTutorialStore.getState();

  if (!draggedPiece || !dragStartPosition) {
    handleBallDragEnd();
    return;
  }

  if (isPositionValidMovementTarget(position)) {
    // Valid drop - move the piece
    movePiece(draggedPiece, position);

    // Handle ball pickup if needed
    const boardSquare = getBoardSquareHelper(
      position,
      useTutorialStore.getState().boardLayout,
    );
    const isPickingUpBall = boardSquare === "ball";

    if (isPickingUpBall) {
      useTutorialStore.setState({
        awaitingDirectionSelection: true,
        selectedPiece: draggedPiece,
        isDragging: false,
        draggedPiece: null,
        dragStartPosition: null,
      });
    } else {
      deselectPiece();
      handleBallDragEnd();
    }

    // Check for step progression
    const { currentStep } = useTutorialStore.getState();
    if (currentStep === "movement_with_ball") {
      nextStep();
    }
  } else {
    // Invalid drop - just end drag
    handleBallDragEnd();
  }
};

/**
 * Handle end of ball drag
 */
export const handleBallDragEnd = () => {
  useTutorialStore.setState({
    isDragging: false,
    draggedPiece: null,
    dragStartPosition: null,
  });
};

/**
 * Handle arrow key direction selection
 * @param direction - The direction to turn the selected piece
 */
export const handleArrowKeyTurn = (direction: FacingDirection): void => {
  const { selectedPiece, awaitingDirectionSelection, currentStep } =
    useTutorialStore.getState();

  if (!selectedPiece) {
    return;
  }

  // If we're in the turning step with a selected piece but not yet awaiting direction,
  // automatically enter direction selection mode
  if (currentStep === "turning" && !awaitingDirectionSelection) {
    useTutorialStore.setState({
      awaitingDirectionSelection: true,
    });
  }

  // Now check if we should proceed with the turn
  const state = useTutorialStore.getState();
  if (!state.awaitingDirectionSelection) {
    return;
  }

  // Get turn targets for the selected piece
  const turnTargets = getTurnTargets(selectedPiece);

  // Find the turn target for this direction
  const turnTarget = turnTargets.find(
    (target) => target.direction === direction,
  );

  if (turnTarget) {
    handleTurnTarget(turnTarget.position);
  }
};

/**
 * Hook to access the tutorial store for reactive updates
 */
export const useTutorialBoard = () => {
  return useTutorialStore();
};
