import React from "react";
import {
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
            "Let's learn how to play this exciting football strategy game. We'll start with the absolute basics - moving a single piece.",
        };

      case "basic_movement":
        return {
          title: "Basic Movement",
          content:
            "Click on the highlighted white piece to select it, then click on any adjacent square (including diagonally) to move it there.",
        };

      case "turning":
        return {
          title: "Turning",
          content: "Click on the piece and attempt to turn it",
        };

      case "movement_with_ball":
        return {
          title: "Movement with Ball",
          content: "Click on any highlighted square to move with the ball.",
        };

      case "passing":
        return {
          title: "Passing",
          content: "Choose a target to pass the ball to.",
        };

      case "completed":
        return {
          title: "Tutorial Complete!",
          content:
            "Congratulations! You've learned the basics of Taka. You're ready to play the full game!",
        };
    }
  };

  const { currentStep, completedSteps } = useTutorialBoard();

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

      <p className="mb-6 leading-relaxed text-gray-600">
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
      </div>

      {/* Completed steps indicator */}
      <div className="mt-6 flex justify-center gap-2">
        {stepOrder.map((step) => (
          <div
            key={step}
            className={`h-3 w-3 rounded-full ${
              completedSteps.has(step as TutorialStep)
                ? "bg-green-500"
                : currentStep === step
                  ? "bg-blue-500"
                  : "bg-gray-300"
            }`}
          />
        ))}
      </div>
    </div>
  );
};

export default TutorialPanel;
