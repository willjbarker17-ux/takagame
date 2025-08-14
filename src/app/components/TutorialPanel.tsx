import React from "react";
import {
  handleRetry,
  nextStep,
  stepOrder,
  useTutorialBoard,
} from "@/hooks/useTutorialStore";
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
            "Let's learn how to play this exciting football strategy game. Click 'Start Learning' to begin.",
        };

      case "turning":
        return {
          title: "Turning (Facing Direction)",
          content:
            "Every time the ball is on a new square with a piece you decide the direction the piece is facing. To do this select the piece with the ball, then click which way you want it to face. You can also use arrow keys (↑↓←→) to turn. ",
        };

      case "basic_movement":
        return {
          title: "Basic Movement",
          content:
            "Click the white piece, then click a highlighted destination square to move it.\n\nYou can move up to three squares forward and two sideways or backward.",
        };

      case "movement_with_ball":
        return {
          title: "Dribbling (Movement with Ball)",
          content:
            "Drag the piece with the ball to an adjacent square to dribble.",
        };

      case "passing":
        return {
          title: "Passing Rules",
          content:
            "Select your piece with the ball, then click on a highlighted teammate to pass. You can only pass in the direction you are facing. After the pass, don't forget to choose the facing direction of the piece.",
        };

      case "consecutive_pass":
        return {
          title: "Consecutive Passes",
          content:
            "Make your first pass, then immediately make a second pass with the receiving piece. This counts as one turn. You cannot chip pass twice when consecutive passing.",
        };

      case "ball_empty_square":
        return {
          title: "Passing to Empty Squares (Loose Ball)",
          content:
            "Select your piece with the ball, then click on an empty square to pass there.",
        };

      case "ball_pickup":
        return {
          title: "Ball Pickup (Gaining Possession)",
          content:
            "Click the piece, then click on the ball to move there and pick it up.",
        };

      case "receiving_passes":
        return {
          title: "Receiving Passes (Moving to Pick Up)",
          content:
            "Pass to an empty square one square away from a teammate. You can move your piece one square to receive the ball.",
        };

      case "chip_pass":
        return {
          title: "Chip Pass (Passing Over Pieces)",
          content:
            "Select your piece with the ball, then pass to a highlighted teammate to pass over defenders. You can pass over pieces as long as they are not directly in front of the direction you are passing.",
        };

      case "shooting":
        return {
          title: "Shooting (Pass to Goal)",
          content:
            "Select your piece with the ball, then click on a highlighted goal square to shoot!",
        };

      case "tackling":
        return {
          title: "Tackling (Stealing the Ball)",
          content:
            "You can tackle (steal the ball from) an opponent if your piece is adjacent (vertically, horizontally, or diagonally) to their piece with the ball. You cannot tackle from behind.\n\nAfter a successful tackle you swap positions and get to make a move.",
        };

      case "activating_goalies":
        return {
          title: "Activating Goalies (Special Pieces)",
          content:
            "Goalies are normal pieces but are the only ones allowed to enter the goal.\n\nGoalies block shots when they are on the square the opponents trying to shoot in. They have no blocking ability when they are in their starting circle ",
        };

      case "blocking_shots":
        return {
          title: "Blocking Shots (Defensive Positioning)",
          content:
            "Now let's practice using your goalie defensively!\n\nThe black piece on the board has the ball and can shoot at your goal. You need to position your goalie to block the shot.\n\nClick the unactivated goalie at the intersection, then place it on position (B5) to block the direct line between the shooter and your goal.",
        };

      case "completed":
        return {
          title: "Tutorial Complete!",
          content:
            "Congratulations! You've learned the basics of Taka. You're ready to play the full game!",
        };
    }
  };

  const { currentStep, completedSteps, showRetryButton, awaitingDirectionSelection } = useTutorialBoard();

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

      {awaitingDirectionSelection && (
        <div className="mb-4 rounded-lg bg-yellow-50 border border-yellow-200 p-3">
          <p className="text-sm text-yellow-800 font-medium">
            💡 Don't forget to choose the piece's direction!
          </p>
        </div>
      )}

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
