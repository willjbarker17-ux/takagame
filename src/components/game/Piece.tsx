import React from "react";
import { Piece as PieceClass } from "@/classes/Piece";
import { FacingDirection } from "@/types/types";
import {
  handleBallDragStart as handleTutorialBallDragStart,
  useTutorialBoard,
} from "@/hooks/useTutorialStore";
import {
  handleBallDragStart as handleGameBallDragStart,
} from "@/hooks/useGameStore";

interface PieceProps {
  piece: PieceClass;
  isSelected: boolean;
  isPassTarget: boolean;
  isTackleTarget?: boolean;
  isDragging?: boolean;
  isOffside?: boolean;
  mode?: "tutorial" | "game"; // Add mode prop to distinguish context
}

const Piece: React.FC<PieceProps> = ({
  piece,
  isSelected,
  isPassTarget,
  isTackleTarget = false,
  isDragging = false,
  isOffside = false,
  mode = "tutorial", // Default to tutorial for backward compatibility
}) => {
  const isGoalie = piece.getIsGoalie();

  // Soccer ball component that positions based on facing direction
  const SoccerBallIcon: React.FC<{ direction: FacingDirection }> = ({
    direction,
  }) => {
    let ballClass = "absolute text-sm pointer-events-none ";

    // Use "right" as default direction if no direction is specified
    const facingDirection = direction || "right";

    // Position ball based on facing direction - at the edge of the piece square
    switch (facingDirection) {
      case "north":
        ballClass += "top-0 left-1/2 transform -translate-x-1/2";
        break;
      case "south":
        ballClass += "bottom-0 left-1/2 transform -translate-x-1/2";
        break;
      case "west":
        ballClass += "left-0 top-1/2 transform -translate-y-1/2";
        break;
      case "east":
        ballClass += "right-0 top-1/2 transform -translate-y-1/2";
        break;
    }

    return <div className={`${ballClass} pointer-events-none`}>⚽</div>;
  };

  const canBeTackled = false;
  const hasBall = piece.getHasBall();
  const tutorialState = useTutorialBoard();
  
  // Only get currentStep if we're in tutorial mode
  const currentStep = mode === "tutorial" ? tutorialState.currentStep : null;

  const handleMouseDown = (e: React.MouseEvent) => {
    e.preventDefault();
    // Use appropriate handler based on mode
    if (mode === "tutorial") {
      handleTutorialBallDragStart(piece, e.clientX, e.clientY);
    } else {
      handleGameBallDragStart(piece, e.clientX, e.clientY);
    }
  };

  return (
    <div
      className={`pointer-events-none relative ${isDragging ? "opacity-30" : ""}`}
    >
      <div
        onMouseDown={handleMouseDown}
        className={`h-10 w-10 rounded-full border-2 shadow-md transition-all duration-200 ${hasBall && (mode === "game" || (mode === "tutorial" && (currentStep === "movement_with_ball" || currentStep === "ball_pickup" || currentStep === "passing" || currentStep === "consecutive_pass" || currentStep === "chip_pass" || currentStep === "shooting" || currentStep === "consecutive_pass_to_score" || currentStep === "offside" || currentStep === "shooting_zone_pass"))) ? "cursor-grab active:cursor-grabbing" : ""} ${
          piece.getColor() === "white"
            ? "border-gray-400 bg-white"
            : "border-gray-600 bg-gray-900"
        } ${
          isSelected && "ring-opacity-75 scale-110 ring-4 ring-yellow-400"
        } ${canBeTackled ? "ring-opacity-75 ring-4 ring-red-500" : ""} ${
          isGoalie ? "!border-4 !border-blue-600 shadow-lg" : ""
        } ${
          isOffside ? "ring-opacity-90 opacity-75 ring-2 ring-orange-500" : ""
        } ${
          isPassTarget
            ? "ring-opacity-75 animate-pulse ring-4 ring-green-500"
            : ""
        } ${
          isTackleTarget
            ? "ring-opacity-75 animate-pulse ring-4 ring-red-500"
            : ""
        }`}
        style={{ pointerEvents: hasBall ? "auto" : "none" }}
      />

      {/* Soccer ball positioned based on facing direction */}
      {piece.getHasBall() && (
        <SoccerBallIcon direction={piece.getFacingDirection()} />
      )}

      {/* Offside indicator */}
      {isOffside && (
        <div className="pointer-events-none absolute -top-1 -right-1 text-xs">
          <div className="flex h-4 w-4 items-center justify-center rounded-full border border-orange-600 bg-orange-500 text-xs font-bold text-white shadow-sm">
            ⚠
          </div>
        </div>
      )}
    </div>
  );
};

export default Piece;
