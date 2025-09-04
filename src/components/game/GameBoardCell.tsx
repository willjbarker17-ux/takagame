"use client";

import React from "react";
import Piece from "./Piece";
import { Piece as PieceClass } from "@/classes/Piece";
import { Position } from "@/classes/Position";
import { BoardSquareType, SquareInfoType } from "@/types/types";
import { BOARD_COLS } from "@/utils/constants";
import { handleSquareClick } from "@/hooks/useGameStore";

interface BoardCellProps {
  position: Position;
  piece: BoardSquareType;
  isOffside: boolean;
  squareInfo: SquareInfoType;
  selectedPiece: PieceClass | null;
  rowIndex: number;
  colIndex: number;
  isDragging?: boolean;
}

const GameBoardCell: React.FC<BoardCellProps> = ({
  position,
  piece,
  isOffside,
  squareInfo,
  selectedPiece,
  rowIndex,
  colIndex,
  isDragging = false,
}) => {
  const cellIndex = rowIndex * BOARD_COLS + colIndex;
  const isGoalCol = colIndex >= 3 && colIndex <= 6;
  const isTopGoal = rowIndex === 0 && isGoalCol;
  const isBottomGoal = rowIndex === 13 && isGoalCol;
  const isGoalLeftCol = colIndex === 3;
  const isGoalRightCol = colIndex === 6;

  const isFieldDivider1 = rowIndex === 4;
  const isFieldDivider2 = rowIndex === 8;
  let borderClasses = `aspect-square border-[0.5px] border-white bg-green-700 flex items-center justify-center relative transition-colors ${squareInfo.clickable ? "cursor-pointer" : "cursor-default"}`;

  if (isFieldDivider1 || isFieldDivider2) {
    borderClasses += " border-b-4 border-b-white";
  }

  if (isTopGoal) {
    borderClasses += " border-b-4 border-b-white";
    if (isGoalLeftCol) borderClasses += " border-l-4 border-b-white";
    if (isGoalRightCol) borderClasses += " border-r-4 border-b-white";
  } else if (isBottomGoal) {
    borderClasses += " border-t-4 border-t-white";
    if (isGoalLeftCol) borderClasses += " border-l-4 border-t-white";
    if (isGoalRightCol) borderClasses += " border-r-4 border-t-white";
  }

  return (
    <div
      key={`cell-${cellIndex}`}
      className={borderClasses}
      data-position={`${position.getPositionCoordinates()[0]},${position.getPositionCoordinates()[1]}`}
      onClick={() => handleSquareClick(position)}
    >
      {/* Render piece with reduced opacity if turn target */}
      {piece !== null && piece instanceof PieceClass && (
        <div
          className={squareInfo.visual === "turn_target" ? "opacity-30" : ""}
        >
          <Piece
            piece={piece}
            isSelected={
              selectedPiece && selectedPiece.getPosition() instanceof Position
                ? selectedPiece
                    .getPositionOrThrowIfUnactivated()
                    .equals(position)
                : false
            }
            isPassTarget={squareInfo.visual === "pass_target"}
            isTackleTarget={squareInfo.visual === "tackle_target"}
            isDragging={isDragging}
            isOffside={isOffside}
            mode="game"
          />
        </div>
      )}

      {/* Turn target indicator */}
      {squareInfo.visual === "turn_target" && selectedPiece && (
        <div className="pointer-events-none absolute inset-0 z-30 flex items-center justify-center">
          <div className="animate-pulse text-2xl text-yellow-400">
            {(() => {
              const [selectedRow, selectedCol] = selectedPiece
                .getPositionOrThrowIfUnactivated()
                .getPositionCoordinates();
              const [currentRow, currentCol] =
                position.getPositionCoordinates();

              // Determine direction based on relative position to selected piece
              if (currentRow < selectedRow) return "↑"; // North
              if (currentRow > selectedRow) return "↓"; // South
              if (currentCol < selectedCol) return "←"; // West
              if (currentCol > selectedCol) return "→"; // East

              return "↻"; // Fallback
            })()}
          </div>
        </div>
      )}

      {squareInfo.visual === "movement" && piece !== "ball" && (
        <div className="pointer-events-none absolute inset-0 z-10 flex items-center justify-center">
          <div className="bg-opacity-90 h-5 w-5 animate-pulse rounded-full border-2 border-blue-600 bg-blue-400 shadow-md" />
        </div>
      )}

      {squareInfo.visual === "empty_pass_target" && (
        <div className="pointer-events-none absolute inset-0 z-10 flex items-center justify-center">
          <div className="h-3 w-3 rounded-full bg-gray-300 shadow-md" />
        </div>
      )}

      {piece === "ball" && (
        <div className="pointer-events-none absolute inset-0 z-20 flex items-center justify-center">
          <div className="relative text-lg">
            ⚽
            {squareInfo.visual === "movement" && (
              <div className="absolute top-1/2 left-1/2 h-4 w-4 -translate-x-1/2 -translate-y-1/2 animate-pulse rounded-full bg-blue-400 opacity-60 mix-blend-multiply" />
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default GameBoardCell;
