import React from "react";
import { handleRetry, nextStep, stepOrder, useTutorialBoard } from "@/hooks/useTutorialStore";
import { TutorialStep } from "@/types/types";

interface TutorialPanelProps {
  className?: string;
}

const TutorialPanel: React.FC<TutorialPanelProps> = ({ className = "" }) => {
  const getTutorialContent = (
    step: TutorialStep,
  ): { title: string; content: string } => {
    switch (step) {
      case "welcome":
        return {
          title: "Welcome to Taka!",
          content:
            "Let's learn how to play this exciting football strategy game. We'll start with the absolute basics - moving a single piece.",
        };

      case "basic_movement":
        return {
          title: "Basic Movement",
          content:
            "Pieces without the ball can move in a single straight line (vertical, horizontal, or diagonal) per turn.\n\nMovement ranges:\n• Forward movement (toward opponent's goal): up to 3 squares\n• Backward movement (toward own goal): up to 2 squares\n• Horizontal movement: up to 2 squares in either direction\n\nPieces cannot move through squares occupied by other pieces.\n\nClick the white piece, then click a valid destination square.",
        };

      case "turning":
        return {
          title: "Turning (Facing Direction)",
          content:
            "Pieces with the ball have a 'facing direction' that determines where they can pass and affects tackling vulnerability.\n\nThe four directions are:\n• Opponent's Goal\n• Own Goal\n• Left (toward Row A)\n• Right (toward Row J)\n\nA piece can 'Turn on the Spot' to change its facing direction without moving - this counts as a complete turn.\n\nSelect the piece with the ball, then click 'Turn Piece' to see direction options.",
        };

      case "movement_with_ball":
        return {
          title: "Dribbling (Movement with Ball)",
          content:
            "A piece with the ball can 'dribble' by moving exactly one square in any direction (vertical, horizontal, or diagonal) to an adjacent empty square.\n\nKey rules:\n• Only 1 square movement (more restrictive than normal 2-3 squares)\n• Must be to an adjacent empty square\n• After dribbling, you must set the piece's new facing direction\n\nDribbling and setting direction completes your turn.,
        };

      case "passing":
        return {
          title: "Passing Rules",
          content:
            "Passes travel in straight lines (vertical, horizontal, diagonal) and are restricted by facing direction.\n\nPassing zones (180-degree cone from front):\n• Facing opponent's goal: forward and forward-diagonally\n• Facing left/right: sideways and diagonally in that direction\n• Facing own goal: cannot pass (must turn first)\n\nSelect your piece with the ball, then click on a highlighted teammate within your passing zone."
        };

      case "consecutive_pass":
        return {
          title: "Consecutive Passes",
          content:
            "You can make two passes in a single turn under specific conditions:\n\n1. First pass goes from Piece A to stationary Piece B\n   (Piece B cannot move to receive)\n\n2. Piece B immediately makes a second pass to Piece C or empty square\n   (facing direction determined by pass path)\n\n3. Final receiver (Piece C) cannot move to receive the second pass\n\nThe entire Pass A→B→C sequence counts as one turn, allowing quick ball movement through multiple teammates."
        };

      case "ball_empty_square":
        return {
          title: "Passing to Empty Squares (Loose Ball)",
          content:
            "You can pass to empty squares within your passing zone, creating a 'loose ball'. The ball remains on that square until a piece moves onto it.\n\nStrategic uses:\n• Positioning the ball away from opponents\n• Setting up plays\n• When no direct passing targets are available\n\nAny piece that moves onto a loose ball square automatically gains possession but must wait until the next turn to act with the ball.",
        };

      case "ball_pickup":
        return {
          title: "Ball Pickup (Gaining Possession)",
          content:
            "When the ball is loose on an empty square, any of your pieces can move there to gain possession.\n\nKey points:\n• Piece follows normal movement rules (2-3 squares depending on direction)\n• Upon arriving at the ball's square, possession is automatic\n• Turn ends immediately upon pickup\n• Piece now has the ball but must wait until your next turn to act\n\nClick the piece, then click on the ball to move there and pick it up.",
        };

      case "receiving_passes":
        return {
          title: "Receiving Passes (Moving to Pick Up)",
          content:
            "When you pass the ball to an empty square, a teammate can move to pick it up if they are within one square of where the ball lands.\n\nKey mechanics:\n• Pass the ball to an empty square\n• The ball must land within one square (adjacent) of a friendly piece\n• That piece can then move onto the ball's square to pick it up\n• The receiving piece gains possession and the turn ends\n• You will need to set the piece's facing direction after pickup\n\nThis allows strategic positioning by passing near teammates rather than directly to them.",
        };

      case "chip_pass":
        return {
          title: "Chip Pass (Passing Over Pieces)",
          content:
            "Chip passes can travel over any number of pieces on their way to the target.\n\nBlocking rules:\n• A chip pass is BLOCKED if an opponent occupies the square that is both immediately adjacent to the passer AND on the direct line of the pass\n\nRestrictions:\n• Cannot make two consecutive chip passes in the same turn\n• Follow same facing direction restrictions as normal passes\n\nThis allows you to pass over defenders to reach teammates behind them."
        };

      case "shooting":
        return {
          title: "Shooting (Special Pass to Goal)",
          content:
            "Shooting is a special type of pass subject to all passing rules.\n\nShooting Zone requirements:\n• White team: columns 10-14 (5 columns closest to black goal)\n• Black team: columns 1-5 (5 columns closest to white goal)\n\nGoal Zone squares:\n• White's goal: D1, E1, F1, G1\n• Black's goal: D14, E14, F14, G14\n\nGoalies can block shots if positioned on the straight-line path from shooter to goal.\n\nShoot at a goal square!"
        };

      case "tackling":
        return {
          title: "Tackling (Stealing the Ball)",
          content:
            "You can tackle (steal the ball from) an opponent if:\n\n1. Your piece is adjacent (vertically, horizontally, or diagonally) to their piece with the ball\n\n2. You're positioned in front of or to the side of their piece relative to their facing direction\n   (You cannot tackle from behind)\n\nAfter a successful tackle:\n• The two pieces swap positions\n• You gain possession of the ball\n• Your piece's facing direction is automatically set based on tackle direction\n\nSelect your white piece and click the highlighted opponent!",
        };

      case "activating_goalies":
        return {
          title: "Activating Goalies (Special Pieces)",
          content:
            "Goalies are special pieces with unique abilities:\n\n1. Only goalies can enter their own Goal Zone squares (4 squares of your goal area)\n\n2. Outside the Goal Zone, they move and act like normal pieces\n\n3. Goalies automatically block shots on goal if positioned on the straight-line path from shooter to goal (no action required)\n\n4. Opponents cannot chip pass over goalies when shooting from the last row (Row A or J)\n\nClick the unactivated goalie at the intersection, then place it in your goal area to activate it.",
        };

      case "completed":
        return {
          title: "Tutorial Complete!",
          content:
            "Congratulations! You've learned the basics of Taka. You're ready to play the full game!",
        };
    }
  };

  const { currentStep, completedSteps, showRetryButton } = useTutorialBoard();

  const tutorialContent = getTutorialContent(currentStep);

  const stepNumber = completedSteps.size + 1;
  const totalSteps = stepOrder.length;

  return (
    <div className={`rounded-lg bg-white p-6 shadow-lg ${className}`}>
      {/* Progress bar */}
      <div className="mb-4">
        <div className="mb-2 flex items-center justify-between">
          <span className="text-sm text-gray-600">Tutorial Progress</span>
          <span className="text-sm text-gray-600">
            {stepNumber}/{totalSteps}
          </span>
        </div>
        <div className="h-2 w-full rounded-full bg-gray-200">
          <div
            className="h-2 rounded-full bg-blue-500 transition-all duration-300"
            style={{ width: `${(stepNumber / totalSteps) * 100}%` }}
          />
        </div>
      </div>

      {/* Content */}
      <h2 className="mb-3 text-2xl font-bold text-gray-800">
        {tutorialContent.title}
      </h2>

      <p className="mb-6 leading-relaxed whitespace-pre-line text-gray-600">
        {tutorialContent.content}
      </p>

      {/* Actions */}
      <div className="flex flex-col gap-3">
        {currentStep === "welcome" && (
          <button
            onClick={nextStep}
            className="w-full cursor-pointer rounded bg-blue-500 px-4 py-2 font-semibold text-white transition-colors hover:bg-blue-600"
          >
            Start Learning
          </button>
        )}
        {showRetryButton && (
          <button
            onClick={handleRetry}
            className="w-full cursor-pointer rounded bg-red-500 px-4 py-2 font-semibold text-white transition-colors hover:bg-red-600"
          >
            Retry
          </button>
        )}
      </div>
    </div>
  );
};

export default TutorialPanel;
