"use client";

import React from "react";
import { BOARD_COLS, BOARD_ROWS } from "@/utils/constants";
import { Position } from "@/classes/Position";
import {
  getSquareInfo,
  handleArrowKeyTurn,
  handleTurnPieceButtonClick,
  handleUnactivatedGoalieClick,
  isCurrentPlayersTurn,
  useGameBoard,
  handleBallDragEnd,
  handleMouseBallDrop,
} from "@/hooks/useGameStore";
import GameBoardCell from "@/components/game/GameBoardCell";
import Piece from "@/components/game/Piece";

const GameBoard: React.FC = () => {
  const {
    boardLayout,
    selectedPiece,
    playerColor,
    whiteUnactivatedGoaliePiece,
    blackUnactivatedGoaliePiece,
    showDirectionArrows,
    playerTurn,
    awaitingConsecutivePass,
    receivingPassPosition,
    isDragging,
    draggedPiece,
  } = useGameBoard();

  const [mousePosition, setMousePosition] = React.useState({ x: 0, y: 0 });
  const [showFloatingPiece, setShowFloatingPiece] = React.useState(false);

  // Handle arrow keys for direction selection
  React.useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Prevent default behavior to avoid scrolling
      const { key } = e;
      if (!["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"].includes(key)) {
        return;
      }

      e.preventDefault();

      // Map arrow keys to directions and call the arrow key handler
      switch (key) {
        case "ArrowUp":
          handleArrowKeyTurn("north");
          break;
        case "ArrowDown":
          handleArrowKeyTurn("south");
          break;
        case "ArrowLeft":
          handleArrowKeyTurn("west");
          break;
        case "ArrowRight":
          handleArrowKeyTurn("east");
          break;
        default:
          return;
      }
    };

    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [showDirectionArrows, selectedPiece]);

  // Handle mouse events for dragging
  React.useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (isDragging) {
        setMousePosition({ x: e.clientX, y: e.clientY });
        // Show floating piece after first mouse move
        setShowFloatingPiece(true);
      }
    };

    const handleMouseUp = (e: MouseEvent) => {
      if (isDragging && draggedPiece) {
        // Temporarily hide the floating piece to detect what's underneath
        const floatingPiece = document.querySelector(
          ".floating-drag-piece",
        ) as HTMLElement;
        const originalDisplay = floatingPiece?.style.display;
        if (floatingPiece) {
          floatingPiece.style.display = "none";
        }

        // Check if we're over a valid drop zone
        const elementUnderMouse = document.elementFromPoint(
          e.clientX,
          e.clientY,
        );

        // Restore the floating piece
        if (floatingPiece) {
          floatingPiece.style.display = originalDisplay || "";
        }

        const boardCell = elementUnderMouse?.closest("[data-position]");

        if (boardCell) {
          const positionData = boardCell.getAttribute("data-position");

          if (positionData) {
            const [row, col] = positionData.split(",").map(Number);
            const position = new Position(row, col);
            
            // Valid drop - execute movement (let the handler validate the move)
            handleMouseBallDrop(position);
            return;
          }
        }
        // Invalid drop - just end drag
        handleBallDragEnd();
      }

      // Reset floating piece visibility
      setShowFloatingPiece(false);
    };

    if (isDragging) {
      document.addEventListener("mousemove", handleMouseMove);
      document.addEventListener("mouseup", handleMouseUp);

      return () => {
        document.removeEventListener("mousemove", handleMouseMove);
        document.removeEventListener("mouseup", handleMouseUp);
      };
    }
  }, [isDragging, draggedPiece]);

  // Reset floating piece visibility when dragging stops
  React.useEffect(() => {
    if (!isDragging) {
      setShowFloatingPiece(false);
    }
  }, [isDragging]);

  // Listen for drag start with initial position
  React.useEffect(() => {
    const handleDragStartPosition = (e: CustomEvent) => {
      const { x, y } = e.detail;
      setMousePosition({ x, y });
      setShowFloatingPiece(true);
    };

    window.addEventListener(
      "dragstart-position",
      handleDragStartPosition as EventListener,
    );
    return () => {
      window.removeEventListener(
        "dragstart-position",
        handleDragStartPosition as EventListener,
      );
    };
  }, []);

  const colLabels = Array.from({ length: BOARD_COLS }, (_, i) =>
    // String.fromCharCode(65 + i),
    i.toString(),
  ); // A-J
  const rowLabels = Array.from({ length: BOARD_ROWS }, (_, i) =>
    // (i + 1).toString(),
    i.toString(),
  ); // 1-14

  const isTurnButtonEnabled = selectedPiece && selectedPiece.getHasBall();

  return (
    <div className="flex h-full w-full flex-col items-center justify-center">
      {/* Turn controls */}
      <div className="mb-4 flex flex-shrink-0 flex-col justify-center gap-4">
        <button
          className={`rounded px-6 py-2 font-semibold text-white transition-colors ${
            !isTurnButtonEnabled
              ? "cursor-not-allowed bg-gray-400"
              : "cursor-pointer bg-blue-600 hover:bg-blue-700"
          }`}
          onClick={handleTurnPieceButtonClick}
          disabled={!isTurnButtonEnabled}
        >
          Turn Piece
        </button>

        <p>
          You are {playerColor} and it&#39;s {playerTurn}&#39;s turn
        </p>
      </div>

      {/* Game board container with shadow and border */}
      <div className="h-full max-h-full w-auto rounded-lg border-2 border-green-200 bg-green-50 p-4 shadow-2xl">
        <div
          className="relative grid h-full"
          style={{
            gridTemplateColumns: `2rem repeat(${BOARD_COLS}, 1fr)`,
            gridTemplateRows: `2rem repeat(${BOARD_ROWS}, 1fr)`,
          }}
        >
          <div className="mb-1 flex items-center justify-center" />

          {colLabels.map((label) => (
            <div
              key={`col-${label}`}
              className="mb-1 flex items-center justify-center text-sm font-semibold text-gray-700"
            >
              {label}
            </div>
          ))}

          {rowLabels.map((rowLabel, rowIndex) => (
            <React.Fragment key={`row-${rowLabel}`}>
              <div className="mr-1 flex items-center justify-center text-sm font-semibold text-gray-700">
                {rowLabel}
              </div>

              {colLabels.map((_, colIndex) => {
                const position = new Position(rowIndex, colIndex);
                let piece = boardLayout[rowIndex][colIndex];

                // Hide the dragged piece from the board during drag
                if (isDragging && draggedPiece && piece === draggedPiece) {
                  piece = null;
                }

                const squareInfo = getSquareInfo(position);

                return (
                  <GameBoardCell
                    key={`cell-${rowIndex}-${colIndex}`}
                    position={position}
                    piece={piece}
                    isOffside={false}
                    squareInfo={squareInfo}
                    selectedPiece={selectedPiece}
                    rowIndex={rowIndex}
                    colIndex={colIndex}
                    isDragging={isDragging}
                  />
                );
              })}
            </React.Fragment>
          ))}

          {/* Render unactivated goalie at intersection */}
          {whiteUnactivatedGoaliePiece && (
            <div
              className={`absolute z-30 flex items-center justify-center ${selectedPiece === whiteUnactivatedGoaliePiece ? "pointer-events-none" : ""}`}
              style={{
                left: `calc(2rem + ${4.25 * (100 / BOARD_COLS) - 0.25}%)`,
                top: `calc(2rem + ${0.4 * (100 / BOARD_ROWS)}%)`,
                width: `${100 / BOARD_COLS}%`,
                height: `${100 / BOARD_ROWS}%`,
              }}
              onClick={() => handleUnactivatedGoalieClick("white")}
            >
              <div
                className={`h-10 w-10 ${playerColor === "white" && isCurrentPlayersTurn() && !awaitingConsecutivePass && !receivingPassPosition ? "cursor-pointer" : ""} rounded-full border-4 border-blue-600 bg-white shadow-md transition-all duration-200 ${selectedPiece === whiteUnactivatedGoaliePiece ? "ring-opacity-75 scale-110 ring-4 ring-yellow-400" : ""}`}
              >
                <div className="flex h-full w-full items-center justify-center text-xs font-bold text-blue-600" />
              </div>
            </div>
          )}

          {/* Render unactivated goalie at intersection */}
          {blackUnactivatedGoaliePiece && (
            <div
              className={`absolute z-30 flex items-center justify-center ${selectedPiece === blackUnactivatedGoaliePiece ? "pointer-events-none" : ""}`}
              style={{
                left: `calc(2rem + ${4.25 * (100 / BOARD_COLS) - 0.25}%)`,
                top: `calc(2rem + ${0.4 * ((100 / BOARD_ROWS) * 30)}%)`,
                width: `${100 / BOARD_COLS}%`,
                height: `${100 / BOARD_ROWS}%`,
              }}
              onClick={() => handleUnactivatedGoalieClick("black")}
            >
              <div
                className={`h-10 w-10 ${playerColor === "black" && isCurrentPlayersTurn() && !awaitingConsecutivePass && !receivingPassPosition ? "cursor-pointer" : ""} rounded-full border-4 border-blue-600 bg-gray-900 shadow-md transition-all duration-200 ${selectedPiece === blackUnactivatedGoaliePiece ? "ring-opacity-75 scale-110 ring-4 ring-yellow-400" : ""}`}
              >
                <div className="flex h-full w-full items-center justify-center text-xs font-bold text-blue-600" />
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Floating drag piece that follows cursor */}
      {isDragging && draggedPiece && showFloatingPiece && (
        <div
          className="floating-drag-piece pointer-events-none fixed z-50"
          style={{
            left: mousePosition.x - 20,
            top: mousePosition.y - 20,
            transform: "scale(1.1)",
            pointerEvents: "none", // Ensure this doesn't interfere with mouse detection
          }}
        >
          <div style={{ pointerEvents: "none" }}>
            <Piece
              piece={draggedPiece}
              isSelected={false}
              isPassTarget={false}
              isTackleTarget={false}
              isOffside={false}
              mode="game"
            />
          </div>
        </div>
      )}
    </div>
  );
};

export default GameBoard;
