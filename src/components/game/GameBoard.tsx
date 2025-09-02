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
} from "@/hooks/useGameStore";
import GameBoardCell from "@/components/game/GameBoardCell";

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
  } = useGameBoard();

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
                const piece = boardLayout[rowIndex][colIndex];

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
    </div>
  );
};

export default GameBoard;
