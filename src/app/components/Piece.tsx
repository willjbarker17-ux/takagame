import React from "react";
import { Piece as PieceClass } from "@/classes/Piece";
import { FacingDirection } from "@/types/types";

interface PieceProps {
  piece: PieceClass;
  isSelected: boolean;
  isPassTarget: boolean;
}

const Piece: React.FC<PieceProps> = ({ piece, isSelected, isPassTarget }) => {
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

    return <div className={ballClass}>⚽</div>;
  };

  const canBeTackled = false;
  const isGoalie = false;
  const isOffside = false;

  return (
    <div className="pointer-events-none relative">
      <div
        className={`h-10 w-10 rounded-full border-2 transition-all duration-200 ${
          piece.getColor() === "white"
            ? "border-gray-400 bg-white shadow-md"
            : "border-gray-600 bg-gray-900 shadow-md"
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
        }`}
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
