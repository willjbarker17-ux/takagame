"use client";

import React from "react";
import Piece from "./Piece";
import { Piece as PieceClass } from "@/classes/Piece";
import {
  BOARD_COLS,
  BOARD_ROWS,
  TUTORIAL_PLAYER_COLOR,
} from "@/utils/constants";
import { Position } from "@/classes/Position";
import {
  getSquareInfo,
  handleSquareClick,
  handleTurnPiece,
  useTutorialBoard,
} from "@/hooks/useTutorialStore";

const TutorialGameBoard: React.FC = () => {
  const { boardLayout, selectedPiece, isTurnButtonEnabled } =
    useTutorialBoard();

  const colLabels = Array.from({ length: BOARD_COLS }, (_, i) =>
    // String.fromCharCode(65 + i),
    i.toString(),
  ); // A-J
  const rowLabels = Array.from({ length: BOARD_ROWS }, (_, i) =>
    // (i + 1).toString(),
    i.toString(),
  ); // 1-14

  return (
    <div className="flex h-full w-full flex-col items-center justify-center">
      {/* Turn controls */}
      <div className="mb-4 flex flex-shrink-0 justify-center gap-4">
        <button
          className={`rounded px-6 py-2 font-semibold text-white transition-colors ${
            !isTurnButtonEnabled
              ? "cursor-not-allowed bg-gray-400"
              : "cursor-pointer bg-blue-600 hover:bg-blue-700"
          }`}
          onClick={handleTurnPiece}
          disabled={!isTurnButtonEnabled}
        >
          Turn Piece
        </button>
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
                const cellIndex = rowIndex * BOARD_COLS + colIndex;
                const isGoalCol = colIndex >= 3 && colIndex <= 6;
                const isTopGoal = rowIndex === 0 && isGoalCol;
                const isBottomGoal = rowIndex === 13 && isGoalCol;
                const isGoalLeftCol = colIndex === 3;
                const isGoalRightCol = colIndex === 6;

                const isFieldDivider1 = rowIndex === 4;
                const isFieldDivider2 = rowIndex === 8;

                const position = new Position(rowIndex, colIndex);

                const piece = boardLayout[rowIndex][colIndex];
                const squareInfo = getSquareInfo(
                  position,
                  TUTORIAL_PLAYER_COLOR,
                );

                let borderClasses = `aspect-square border-[0.5px] border-white bg-green-700 flex items-center justify-center relative transition-colors ${squareInfo === "nothing" ? "cursor-default" : "cursor-pointer"}`;

                if (isFieldDivider1 || isFieldDivider2) {
                  borderClasses += " border-b-4 border-b-white";
                }

                if (isTopGoal) {
                  borderClasses += " border-b-4 border-b-white";
                  if (isGoalLeftCol)
                    borderClasses += " border-l-4 border-b-white";
                  if (isGoalRightCol)
                    borderClasses += " border-r-4 border-b-white";
                } else if (isBottomGoal) {
                  borderClasses += " border-t-4 border-t-white";
                  if (isGoalLeftCol)
                    borderClasses += " border-l-4 border-t-white";
                  if (isGoalRightCol)
                    borderClasses += " border-r-4 border-t-white";
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
                        <div className="animate-pulse text-2xl text-yellow-400">
                          ↻
                        </div>
                      </div>
                    ) : (
                      <>
                        {piece !== null && piece instanceof PieceClass && (
                          <Piece
                            piece={piece}
                            isSelected={
                              selectedPiece?.getPosition()?.equals(position) ??
                              false
                            }
                            isPassTarget={squareInfo === "pass_target"}
                          />
                        )}

                        {piece === "ball" && (
                          <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
                            <div className="text-lg">⚽</div>
                          </div>
                        )}

                        {/*{isOverlappingPosition(rowIndex, colIndex) && (*/}
                        {/*  <div className="pointer-events-none absolute inset-0 flex items-center justify-center">*/}
                        {/*    <div className="bg-opacity-90 h-6 w-6 animate-pulse rounded-full border-2 border-yellow-500 bg-gradient-to-r from-white to-blue-400 shadow-md" />*/}
                        {/*  </div>*/}
                        {/*)}*/}

                        {/*{isDribbleOnlyPosition(rowIndex, colIndex) && (*/}
                        {/*  <div className="pointer-events-none absolute inset-0 flex items-center justify-center">*/}
                        {/*    <div className="bg-opacity-80 h-4 w-4 rounded-full border border-gray-300 bg-gray-200 shadow-sm" />*/}
                        {/*  </div>*/}
                        {/*)}*/}

                        {squareInfo === "movement" && (
                          <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
                            <div className="bg-opacity-90 h-5 w-5 animate-pulse rounded-full border-2 border-blue-600 bg-blue-400 shadow-md" />
                          </div>
                        )}

                        {/*{isShootOnlyPosition(rowIndex, colIndex) && (*/}
                        {/*  <div className="pointer-events-none absolute inset-0 flex items-center justify-center">*/}
                        {/*    <div className="bg-opacity-90 h-5 w-5 animate-pulse rounded-full border-2 border-red-700 bg-red-500 shadow-md" />*/}
                        {/*  </div>*/}
                        {/*)}*/}
                      </>
                    )}
                  </div>
                );
              })}
            </React.Fragment>
          ))}

          {/* Inactive goalies */}
          {/*{pieces.find(*/}
          {/*  (p) =>*/}
          {/*    p.type === "goalie" &&*/}
          {/*    p.color === "white" &&*/}
          {/*    p.isActive === false,*/}
          {/*) && (*/}
          {/*  <div*/}
          {/*    className="pointer-events-auto absolute z-10"*/}
          {/*    style={{*/}
          {/*      left: "calc(2rem + 1.0 * (100% - 2rem) / 14)",*/}
          {/*      top: "calc(2rem + 5.0 * (100% - 2rem) / 10)",*/}
          {/*      transform: "translate(-50%, -50%)",*/}
          {/*    }}*/}
          {/*  >*/}
          {/*    <Piece*/}
          {/*      color="white"*/}
          {/*      hasBall={false}*/}
          {/*      isGoalie={true}*/}
          {/*      isSelected={selectedPiece === "white-goalie"}*/}
          {/*      onClick={() => selectPiece("white-goalie")}*/}
          {/*      clickable={currentPlayer === "white" && !tutorialActive}*/}
          {/*      canBeTackled={false}*/}
          {/*      isOffside={false}*/}
          {/*      canReceiveBall={false}*/}
          {/*    />*/}
          {/*  </div>*/}
          {/*)}*/}

          {/*{pieces.find(*/}
          {/*  (p) =>*/}
          {/*    p.type === "goalie" &&*/}
          {/*    p.color === "black" &&*/}
          {/*    p.isActive === false,*/}
          {/*) && (*/}
          {/*  <div*/}
          {/*    className="pointer-events-auto absolute z-10"*/}
          {/*    style={{*/}
          {/*      left: "calc(2rem + 13.0 * (100% - 2rem) / 14)",*/}
          {/*      top: "calc(2rem + 5.0 * (100% - 2rem) / 10)",*/}
          {/*      transform: "translate(-50%, -50%)",*/}
          {/*    }}*/}
          {/*  >*/}
          {/*    <Piece*/}
          {/*      color="black"*/}
          {/*      hasBall={false}*/}
          {/*      isGoalie={true}*/}
          {/*      isSelected={selectedPiece === "black-goalie"}*/}
          {/*      onClick={() => selectPiece("black-goalie")}*/}
          {/*      clickable={currentPlayer === "black" && !tutorialActive}*/}
          {/*      canBeTackled={false}*/}
          {/*      isOffside={false}*/}
          {/*      canReceiveBall={false}*/}
          {/*    />*/}
          {/*  </div>*/}
          {/*)}*/}
        </div>
      </div>
    </div>
  );
};

export default TutorialGameBoard;
