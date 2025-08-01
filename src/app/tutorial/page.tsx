"use client";

import React from "react";
import TutorialGameBoard from "../components/TutorialGameBoard";
import TutorialPanel from "../components/TutorialPanel";

const DemoPage: React.FC = () => {
  return (
    <div className="flex h-screen flex-col bg-gradient-to-br from-green-50 to-blue-50 p-4">
      <div className="mx-auto flex h-full w-full max-w-7xl flex-col">
        <div className="mb-6 flex-shrink-0 text-center">
          <h1 className="mb-4 text-4xl font-bold text-gray-800">
            Taka - Interactive Tutorial
          </h1>
          <p className="text-lg text-gray-600">
            Learn to play Taka with step-by-step interactive guidance
          </p>
        </div>

        <div className="flex min-h-0 flex-1 flex-col gap-8 lg:flex-row">
          <div className="flex-1 lg:w-3/4">
            <TutorialGameBoard />
          </div>

          <div className="flex-shrink-0 lg:w-1/4">
            <TutorialPanel />
          </div>
        </div>
      </div>
    </div>
  );
};

export default DemoPage;
