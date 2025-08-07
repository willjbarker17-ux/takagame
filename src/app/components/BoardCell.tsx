"use client";

import React from "react";
import Piece from "./Piece";
import { Piece as PieceClass } from "@/classes/Piece";
import { Position } from "@/classes/Position";
import { BoardSquareType, SquareType } from "@/types/types";
import { handleSquareClick } from "@/hooks/useTutorialStore";
import { BOARD_COLS } from "@/utils/constants";

interface BoardCellProps {
  position: Position;
  piece: BoardSquareType;
  squareInfo: SquareType;
  selectedPiece: PieceClass | null;
  rowIndex: number;
  colIndex: number;
}

const BoardCell: React.FC<BoardCellProps> = ({
  position,
  piece,
  squareInfo,
  selectedPiece,
  rowIndex,
  colIndex,
}) => {
  const cellIndex = rowIndex * BOARD_COLS + colIndex;
  const isGoalCol = colIndex >= 3 && colIndex <= 6;
  const isTopGoal = rowIndex === 0 && isGoalCol;
  const isBottomGoal = rowIndex === 13 && isGoalCol;
  const isGoalLeftCol = colIndex === 3;
  const isGoalRightCol = colIndex === 6;

  const isFieldDivider1 = rowIndex === 4;
  const isFieldDivider2 = rowIndex === 8;

  let borderClasses = `aspect-square border-[0.5px] border-white bg-green-700 flex items-center justify-center relative transition-colors ${squareInfo === "nothing" ? "cursor-default" : "cursor-pointer"}`;

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
      onClick={() => handleSquareClick(position)}
    >
      {/* If this square is a turn target, then that is all we care about */}
      {squareInfo === "turn_target" ? (
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
          <div className="animate-pulse text-2xl text-yellow-400">↻</div>
        </div>
      ) : (
        <>
          {piece !== null && piece instanceof PieceClass && (
            <Piece
              piece={piece}
              isSelected={
                selectedPiece
                  ?.getPositionOrThrowIfUnactivated()
                  ?.equals(position) ?? false
              }
              isPassTarget={squareInfo === "pass_target"}
              isTackleTarget={squareInfo === "tackle_target"}
            />
          )}

          {squareInfo === "movement" && piece !== "ball" && (
            <div className="pointer-events-none absolute inset-0 z-10 flex items-center justify-center">
              <div className="bg-opacity-90 h-5 w-5 animate-pulse rounded-full border-2 border-blue-600 bg-blue-400 shadow-md" />
            </div>
          )}

          {piece === "ball" && (
            <div className="pointer-events-none absolute inset-0 z-20 flex items-center justify-center">
              <div className="relative text-lg">
                ⚽
                {squareInfo === "movement" && (
                  <div className="absolute top-1/2 left-1/2 h-4 w-4 -translate-x-1/2 -translate-y-1/2 animate-pulse rounded-full bg-blue-400 opacity-60 mix-blend-multiply" />
                )}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
};

export default BoardCell;
