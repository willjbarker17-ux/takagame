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
  PiecePositionType,
  PlayerColor,
  SquareInfoType,
} from "@/types/types";
import { create } from "zustand";
import {
  addPieceToBoard,
  createBoardLayout,
  getAdjacentPieces,
  getAdjacentPositions,
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
import { GameClient } from "@/services/socketService";


interface Player {
  id: string;
  username: string;
  elo: number;
  isGuest?: boolean;
}

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
  /** If we are trying to receive a pass, we put the position that the ball was passed to here */
  receivingPassPosition: Position | null;
  /** Drag state for ball movement */
  isDragging: boolean;
  draggedPiece: Piece | null;
  dragStartPosition: Position | null;

  // Multiplayer state
  /** Socket client instance for multiplayer communication */
  socketClient: GameClient | null;
  /** Is the socket connected? */
  isConnected: boolean;
  /** Is currently connecting to socket? */
  isConnecting: boolean;
  /** Connection/socket error message */
  connectionError: string | null;
  /** Game status from backend */
  gameStatus: "waiting" | "active" | "completed";
  /** White player information */
  whitePlayer: Player | null;
  /** Black player information */
  blackPlayer: Player | null;
  /** Winner of the game */
  winner: PlayerColor | null;
  /** Waiting for opponent to join */
  waitingForOpponent: boolean;
}

// Initial state is empty - pieces will be loaded from multiplayer game state

export const useGameStore = create<GameState>(() => ({
  gameId: "",
  pieces: [],
  balls: [],
  boardLayout: createBoardLayout([], []),
  selectedPiece: null,
  playerColor: "white",
  playerTurn: "white",
  whiteUnactivatedGoaliePiece: null,
  blackUnactivatedGoaliePiece: null,
  showDirectionArrows: false,
  isSelectionLocked: false,
  awaitingConsecutivePass: false,
  receivingPassPosition: null,
  isDragging: false,
  draggedPiece: null,
  dragStartPosition: null,

  // Multiplayer state
  socketClient: null,
  isConnected: false,
  isConnecting: false,
  connectionError: null,
  gameStatus: "waiting",
  whitePlayer: null,
  blackPlayer: null,
  winner: null,
  waitingForOpponent: true,
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
    awaitingConsecutivePass,
    receivingPassPosition,
  } = useGameStore.getState();

  if (color !== playerColor || awaitingConsecutivePass || receivingPassPosition)
    return;

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

  // If we are waiting to receive a pass, and we haven't selected a piece yet, only allow the user to select adjacent pieces
  if (state.receivingPassPosition && !state.selectedPiece) {
    if (!pieceAtPosition) return { visual: "nothing", clickable: false };

    const adjPieces = getAdjacentPieces(
      state.receivingPassPosition,
      state.playerColor,
      state.boardLayout,
    );

    const piece = adjPieces.find((p) => p === pieceAtPosition);

    if (!piece || piece.getColor() !== state.playerColor) {
      return { visual: "nothing", clickable: false };
    }

    return { visual: "piece", clickable: true };
  }

  const playersGoalie = getUnactivatedGoalie();

  if (state.receivingPassPosition && state.selectedPiece) {
    if (state.selectedPiece === playersGoalie) {
      // TODO: Is this right?
      throw new Error("We can't receive by an unactivated goalie");
    }

    // Find all positions adjacent to the selected piece.
    const adjPositions = getAdjacentPositions(
      state.selectedPiece.getPositionOrThrowIfUnactivated(),
    );

    // See if the adjacent position is 1. equal to the current position we are checking and 2. the receiving pass position
    const positionIsAdjToSelectedPiece = adjPositions.find(
      (p) => p.equals(position) && p.equals(state.receivingPassPosition!),
    );

    if (positionIsAdjToSelectedPiece) {
      return { visual: "movement", clickable: true };
    }

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
  // Send the current game state to the server before ending turn
  const { socketClient, pieces, balls } = useGameStore.getState();

  if (socketClient) {
    // Convert local game state to socket format, filtering out unactivated pieces
    const socketGameState = {
      pieces: pieces
        .map((piece) => {
          // Check if this is an unactivated goalie
          let x: number | undefined;
          let y: number | undefined;
          
          try {
            const [row, col] = piece.getPositionOrThrowIfUnactivated().getPositionCoordinates();
            x = col;
            y = row;
          } catch {
            // This is an unactivated piece, don't include it
            return null;
          }

          return {
            id: piece.getId(),
            playerId:
              piece.getColor() === "white"
                ? "white_player_id"
                : "black_player_id",
            type: piece.getIsGoalie() ? "goalie" : "regular",
            hasBall: piece.getHasBall(),
            facingDirection: piece.getFacingDirection(),
            x: x!,
            y: y!,
          };
        })
        .filter((piece) => piece !== null),
      ballPositions: balls.map((ball) => {
        const [row, col] = ball.getPositionCoordinates();
        return {
          x: col, // Backend now uses 0-indexed columns
          y: row, // Backend now uses 0-indexed rows
        };
      }),
    };

    try {
      console.log("Sending move to socket:", socketGameState);
      socketClient.makeMove(socketGameState);
    } catch (error) {
      console.error("Failed to send move:", error);
    }
  }

  useGameStore.setState((state) => {
    return {
      playerTurn: state.playerTurn === "white" ? "black" : "white",
      selectedPiece: null,
      isSelectionLocked: false,
      awaitingConsecutivePass: false,
      receivingPassPosition: null,
      isDragging: false,
      draggedPiece: null,
      dragStartPosition: null,
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
  const { selectedPiece, boardLayout, playerColor } = useGameStore.getState();

  if (!selectedPiece) {
    throw new Error(
      "Trying to pass the ball to an empty square, but there is no selected piece",
    );
  }

  const pieceAtDestination = getPieceAtPosition(position, boardLayout);

  if (pieceAtDestination) {
    throw new Error(
      "We are trying to perform an empty square pass, but there is a piece at the destination",
    );
  }

  // Check for scoring
  if (position.isPositionInGoal()) {
    // Place the ball at the goal position (this will remove ball from piece)
    passBall(selectedPiece.getPositionOrThrowIfUnactivated(), position);

    endTurn();
    return;
  }

  passBall(selectedPiece.getPositionOrThrowIfUnactivated(), position);

  if (
    isCrossZonePass(selectedPiece.getPositionOrThrowIfUnactivated(), position)
  ) {
    // Turn is over if this was a cross zone pass
    endTurn();
    return;
  }

  // Get all adjacent pieces that don't include the currently selected piece
  const adjPieces = getAdjacentPieces(
    position,
    playerColor,
    boardLayout,
  ).filter((e) => e !== selectedPiece);

  // If there are no adjacent pieces, we can't physically receive the pass so end the turn
  if (adjPieces.length === 0) {
    endTurn();
    return;
  }

  useGameStore.setState({
    receivingPassPosition: position,
    selectedPiece: null,
  });
};

/**
 * Pass the ball
 * @param origin Origin piece to pass ball from
 * @param destination Destination to pass ball to
 */
const passBall = (origin: Position, destination: Position) => {
  const { boardLayout, balls } = useGameStore.getState();

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
      balls: [...balls, destination],
    });
  }
};

const handlePassTargetClick = (position: Position): void => {
  const { selectedPiece, boardLayout, awaitingConsecutivePass } =
    useGameStore.getState();
  const pieceAtDestination = getPieceAtPosition(position, boardLayout);

  if (!selectedPiece) {
    throw new Error("Trying to pass but there isn't a selected piece");
  }

  if (!pieceAtDestination) {
    throw new Error(
      "There isn't a piece at the destination, but we were told from getSquareInfo that this position was a pass target.",
    );
  }

  const fromPosition = selectedPiece.getPositionOrThrowIfUnactivated();
  passBall(fromPosition, position);

  // Select the piece that received the pass
  useGameStore.setState({
    selectedPiece: pieceAtDestination,
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
  const { selectedPiece, isDragging } = useGameStore.getState();

  if (!selectedPiece) {
    throw new Error(
      "Attempting to move a piece, but there isn't a selected piece",
    );
  }

  // Pieces with the ball are not allowed to move by click unless during drag
  if (selectedPiece.getHasBall() && !isDragging) {
    return; // Silently ignore click-based movement for pieces with ball
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
 * Handle start of ball drag
 */
export const handleBallDragStart = (
  piece: Piece,
  initialX?: number,
  initialY?: number,
) => {
  const { playerColor, playerTurn } = useGameStore.getState();

  // Only allow dragging during player's turn
  if (playerColor !== playerTurn) return;

  if (piece.getColor() !== playerColor) return;

  if (!piece.getHasBall()) return;

  // Select the piece and initiate drag
  useGameStore.setState({
    selectedPiece: piece,
    isDragging: true,
    draggedPiece: piece,
    dragStartPosition: piece.getPositionOrThrowIfUnactivated(),
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
 * Handle mouse-based ball drop
 */
export const handleMouseBallDrop = (position: Position) => {
  const { draggedPiece, dragStartPosition, boardLayout } =
    useGameStore.getState();

  if (!draggedPiece || !dragStartPosition) {
    handleBallDragEnd();
    return;
  }

  const isValidMovement = isPositionValidMovementTarget(
    draggedPiece,
    position,
    boardLayout,
  );

  if (isValidMovement) {
    // Valid drop - move the piece
    movePiece(draggedPiece, position);

    // End drag state
    handleBallDragEnd();

    endTurn();
    return;
  } else {
    // Invalid drop - just end drag
    handleBallDragEnd();
  }
};

/**
 * Handle end of ball drag
 */
export const handleBallDragEnd = () => {
  useGameStore.setState({
    isDragging: false,
    draggedPiece: null,
    dragStartPosition: null,
  });
};

// Multiplayer socket functions

/**
 * Initialize socket client and set up event handlers
 */
export const initializeSocketClient = (
  client: GameClient,
  currentUserId?: string,
) => {
  useGameStore.setState({
    socketClient: client,
    isConnecting: true,
    connectionError: null,
  });

  // Set up event handlers
  client.setOnConnect(() => {
    useGameStore.setState({
      isConnected: true,
      isConnecting: false,
      connectionError: null,
    });
  });

  client.setOnDisconnect(() => {
    useGameStore.setState({
      isConnected: false,
      connectionError: "Disconnected from server",
    });
  });

  client.setOnGameJoined((data) => {
    const { game } = data;

    // Debug logging for player assignment
    console.log("Player joining game - Game data:", {
      whitePlayerId: game.whitePlayerId,
      blackPlayerId: game.blackPlayerId,
      whitePlayerGuestId: game.whitePlayerGuestId,
      blackPlayerGuestId: game.blackPlayerGuestId,
      whitePlayerUsername: game.whitePlayerUsername,
      blackPlayerUsername: game.blackPlayerUsername,
      isGuestUser: client.isGuestUser(),
      guestSession: client.getGuestSession(),
    });

    // Determine player color based on authenticated or guest user
    let playerColor: PlayerColor = "white"; // Default fallback

    if (client.isGuestUser()) {
      const guestSession = client.getGuestSession();

      // First try to match by session ID (most reliable)
      if (guestSession && guestSession.sessionId) {
        if (game.whitePlayerGuestId === guestSession.sessionId) {
          playerColor = "white";
        } else if (game.blackPlayerGuestId === guestSession.sessionId) {
          playerColor = "black";
        }
      }

      // If session ID matching failed, try to match by username as fallback
      if (
        playerColor === "white" &&
        guestSession?.sessionId !== game.whitePlayerGuestId
      ) {
        // Reset color and try username matching
        const currentUsername =
          guestSession?.username || client.getGuestUsername();
        if (currentUsername === game.whitePlayerUsername) {
          playerColor = "white";
        } else if (currentUsername === game.blackPlayerUsername) {
          playerColor = "black";
        } else {
          // Final fallback: use game state to infer color
          if (game.whitePlayerGuestId && game.blackPlayerGuestId) {
            // Both slots filled - assume we're the second player (black) if we can't match either ID
            playerColor = "black";
          } else if (game.whitePlayerGuestId && !game.blackPlayerGuestId) {
            playerColor = "black"; // White slot taken, we must be black
          } else if (!game.whitePlayerGuestId && game.blackPlayerGuestId) {
            playerColor = "white"; // Black slot taken, we must be white
          } else {
            playerColor = "white"; // Default to white if neither slot filled
          }
        }
      }
    } else {
      // For authenticated users, use the passed currentUserId or try to determine from context
      if (currentUserId) {
        // Compare current user ID with game player IDs
        if (game.whitePlayerId === currentUserId) {
          playerColor = "white";
        } else if (game.blackPlayerId === currentUserId) {
          playerColor = "black";
        }
      } else {
        // Fallback: infer from game state when user ID is not available
        // This logic assumes the joining user is the opposite color of existing player
        if (game.whitePlayerId && !game.blackPlayerId) {
          // White player exists, joining player must be black
          playerColor = "black";
        } else if (!game.whitePlayerId && game.blackPlayerId) {
          // Black player exists, joining player must be white
          playerColor = "white";
        } else {
          // Default to white for first player
          playerColor = "white";
        }
      }
    }

    // Debug logging for final player color assignment
    console.log("Final player color assignment:", {
      playerColor,
      isGuestUser: client.isGuestUser(),
      guestSessionId: client.getGuestSession()?.sessionId,
      currentUsername:
        client.getGuestSession()?.username || client.getGuestUsername(),
      gameWhiteGuestId: game.whitePlayerGuestId,
      gameBlackGuestId: game.blackPlayerGuestId,
      gameWhiteUsername: game.whitePlayerUsername,
      gameBlackUsername: game.blackPlayerUsername,
    });

    // Create player objects with proper data
    const whitePlayer: Player | null =
      game.whitePlayer ||
      (game.whitePlayerUsername
        ? {
            id: game.whitePlayerGuestId || "guest-white",
            username: game.whitePlayerUsername,
            elo: 1200, // Default ELO for guests
            isGuest: true,
          }
        : null);

    const blackPlayer: Player | null =
      game.blackPlayer ||
      (game.blackPlayerUsername
        ? {
            id: game.blackPlayerGuestId || "guest-black",
            username: game.blackPlayerUsername,
            elo: 1200, // Default ELO for guests
            isGuest: true,
          }
        : null);

    useGameStore.setState({
      gameId: game.id,
      gameStatus: game.status,
      whitePlayer,
      blackPlayer,
      playerColor,
      playerTurn: game.currentTurn || "white",
      waitingForOpponent: game.status === "waiting",
      winner: game.winner,
    });

    if (game.gameState) {
      updateGameStateFromSocket(game.gameState);
    }
  });

  client.setOnPlayerJoined((data) => {
    const { game } = data;

    // Create player objects with proper data for both authenticated and guest users
    const whitePlayer: Player | null =
      game.whitePlayer ||
      (game.whitePlayerUsername
        ? {
            id: game.whitePlayerGuestId || "guest-white",
            username: game.whitePlayerUsername,
            elo: 1200,
            isGuest: true,
          }
        : null);

    const blackPlayer: Player | null =
      game.blackPlayer ||
      (game.blackPlayerUsername
        ? {
            id: game.blackPlayerGuestId || "guest-black",
            username: game.blackPlayerUsername,
            elo: 1200,
            isGuest: true,
          }
        : null);

    useGameStore.setState({
      gameStatus: game.status,
      whitePlayer,
      blackPlayer,
      waitingForOpponent: false,
      playerTurn: game.currentTurn || "white",
    });

    if (game.gameState) {
      updateGameStateFromSocket(game.gameState);
    }
  });

  client.setOnGameStateUpdated((data) => {
    const { game, gameState } = data;
    useGameStore.setState({
      playerTurn: game.currentTurn || "white",
      gameStatus: game.status,
      winner: game.winner,
    });
    updateGameStateFromSocket(gameState);
  });

  client.setOnMoveConfirmed((data) => {
    const { game, gameState } = data;
    useGameStore.setState({
      playerTurn: game.currentTurn || "white",
      gameStatus: game.status,
      winner: game.winner,
    });
    updateGameStateFromSocket(gameState);
  });

  client.setOnGameOver((data) => {
    const { game, winner } = data;
    useGameStore.setState({
      gameStatus: "completed",
      winner: winner as PlayerColor,
      playerTurn: game.currentTurn || "white",
    });
  });

  client.setOnError((error) => {
    useGameStore.setState({
      connectionError: error.message,
      isConnecting: false,
      isConnected: false, // Treat access denied as connection failure
    });
  });
};

// Types for socket data format
interface SocketGamePiece {
  id: string;
  playerId: string;
  x?: number; // Optional for unactivated goalies
  y?: number; // Optional for unactivated goalies
  type?: string;
  hasBall?: boolean;
  facingDirection?: string;
}

interface SocketBallPosition {
  x: number;
  y: number;
}

interface SocketGameState {
  pieces: SocketGamePiece[];
  ballPositions: SocketBallPosition[];
}

/**
 * Update local game state from socket data
 */
const updateGameStateFromSocket = (socketGameState: SocketGameState) => {
  if (
    !socketGameState ||
    !socketGameState.pieces ||
    !socketGameState.ballPositions
  ) {
    console.log("Invalid game state received from socket:", socketGameState);
    return;
  }

  // Convert socket game pieces to local Piece objects
  const pieces: Piece[] = socketGameState.pieces.map(
    (socketPiece: SocketGamePiece) => {
      // Determine player color based on piece ID or playerId
      const color: PlayerColor =
        socketPiece.id?.startsWith("w") ||
        socketPiece.playerId?.includes("white")
          ? "white"
          : "black";

      // Handle unactivated goalies with null/undefined coordinates
      const position = (socketPiece.x === undefined || socketPiece.y === undefined) 
        ? (color === "white" ? "white_unactivated" : "black_unactivated")
        : new Position(socketPiece.y!, socketPiece.x!);

      return new Piece({
        id: socketPiece.id,
        color: color,
        position: position as PiecePositionType,
        hasBall: socketPiece.hasBall || false,
        facingDirection: (socketPiece.facingDirection as FacingDirection) || (color === "white" ? "south" : "north"),
        isGoalie: socketPiece.type === "goalie",
      });
    },
  );

  // Convert socket ball positions to local Position objects
  const balls: Position[] = socketGameState.ballPositions.map(
    (ballPos: SocketBallPosition) => new Position(ballPos.y, ballPos.x), // Both backend and frontend now use 0-indexed
  );

  // Note: Pieces already have correct hasBall property from server data, no need to reassign

  // Filter out balls that are held by pieces (only keep loose balls)
  const looseBalls = balls.filter(
    (ball) =>
      !pieces.some((piece) => {
        try {
          const piecePosition = piece.getPositionOrThrowIfUnactivated();
          return piecePosition.equals(ball);
        } catch {
          return false; // Unactivated piece
        }
      }),
  );

  // Create board layout
  const boardLayout = createBoardLayout(pieces, looseBalls);

  // Check for unactivated goalies
  const whiteUnactivatedGoaliePiece =
    pieces.find(
      (p) =>
        p.getColor() === "white" &&
        p.getIsGoalie() &&
        typeof p.getPosition() === "string",
    ) || null;

  const blackUnactivatedGoaliePiece =
    pieces.find(
      (p) =>
        p.getColor() === "black" &&
        p.getIsGoalie() &&
        typeof p.getPosition() === "string",
    ) || null;

  // Update the store with the converted state
  useGameStore.setState({
    pieces,
    balls: looseBalls,
    boardLayout,
    whiteUnactivatedGoaliePiece,
    blackUnactivatedGoaliePiece,
  });

  console.log("Updated game state from socket:", {
    pieces: pieces.length,
    balls: looseBalls.length,
  });
};

/**
 * Send move to multiplayer backend
 */
export const sendMoveToSocket = () => {
  const { socketClient, pieces, balls } = useGameStore.getState();

  if (!socketClient) {
    console.error("Cannot send move: Socket client not initialized");
    return;
  }

  // Convert local game state to socket format, filtering out unactivated pieces
  const socketGameState = {
    pieces: pieces
      .map((piece) => {
        // Check if this is an unactivated goalie
        let x: number | undefined;
        let y: number | undefined;
        
        try {
          const [row, col] = piece.getPositionOrThrowIfUnactivated().getPositionCoordinates();
          x = col;
          y = row;
        } catch {
          // This is an unactivated piece, don't include it
          return null;
        }

        return {
          id: piece.getId(),
          playerId:
            piece.getColor() === "white" ? "white_player_id" : "black_player_id", // TODO: Use actual player IDs
          type: piece.getIsGoalie() ? "goalie" : "regular",
          hasBall: piece.getHasBall(),
          facingDirection: piece.getFacingDirection(),
          x: x!,
          y: y!,
        };
      })
      .filter((piece) => piece !== null),
    ballPositions: balls.map((ball) => ({
      x: ball.getPositionCoordinates()[0],
      y: ball.getPositionCoordinates()[1],
    })),
  };

  try {
    socketClient.makeMove(socketGameState);
  } catch (error) {
    console.error("Failed to send move:", error);
    useGameStore.setState({
      connectionError:
        error instanceof Error ? error.message : "Failed to send move",
    });
  }
};

/**
 * Connect to game with socket client
 */
export const connectToGame = async (gameId: string) => {
  const { socketClient } = useGameStore.getState();

  if (!socketClient) {
    throw new Error("Socket client not initialized");
  }

  try {
    await socketClient.connect();
    socketClient.joinGame(gameId);
  } catch (error) {
    useGameStore.setState({
      connectionError:
        error instanceof Error ? error.message : "Failed to connect to game",
      isConnecting: false,
    });
    throw error;
  }
};

/**
 * Create a new multiplayer game (supports both guest and authenticated users)
 */
export const createMultiplayerGame = async (
  guestUsername?: string,
): Promise<string> => {
  const { socketClient } = useGameStore.getState();

  if (!socketClient) {
    throw new Error("Socket client not initialized");
  }

  try {
    const gameId = await socketClient.createGame(guestUsername);

    // Store guest session if this was a guest game creation
    if (socketClient.isGuestUser()) {
      const guestSession = socketClient.getGuestSession();
      if (guestSession) {
        // The auth hook will handle storing this via setGuestSession
        console.log("Guest session created:", guestSession);
      }
    }

    return gameId;
  } catch (error) {
    useGameStore.setState({
      connectionError:
        error instanceof Error ? error.message : "Failed to create game",
    });
    throw error;
  }
};

/**
 * Disconnect from multiplayer game
 */
export const disconnectFromGame = () => {
  const { socketClient } = useGameStore.getState();

  if (socketClient) {
    socketClient.disconnect();
  }

  useGameStore.setState({
    socketClient: null,
    isConnected: false,
    isConnecting: false,
    connectionError: null,
    gameStatus: "waiting",
    whitePlayer: null,
    blackPlayer: null,
    winner: null,
    waitingForOpponent: true,
  });
};

/**
 * Hook to access the game store and initialize it with server data
 */
export const useGameBoard = () => {
  return useGameStore();
};
