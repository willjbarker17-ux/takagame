"use client";

import React, { useEffect, useState } from "react";
import FullPageLoader from "@/components/FullPageLoader";
import GameBoard from "@/components/game/GameBoard";
import { useGameBoard } from "@/hooks/useGameStore";
import { Clock, Users, Trophy, Settings } from "lucide-react";

const PlayPage: React.FC = () => {
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [gameId, setGameId] = useState<string | null>(null);
  const [globalError, setGlobalError] = useState<string | null>(null);
  const [gameTime, setGameTime] = useState<number>(0);

  const { playerColor, playerTurn } = useGameBoard();

  useEffect(() => {
    const params = new URLSearchParams(window.location.href);
    const gameId = params.get("id");

    if (gameId) {
      setGameId(gameId);
    }
  }, []);

  // Game timer effect
  useEffect(() => {
    const timer = setInterval(() => {
      setGameTime((prev) => prev + 1);
    }, 1000);

    return () => clearInterval(timer);
  }, []);

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
  };

  if (isLoading) {
    return <FullPageLoader />;
  }

  if (globalError) {
    return <h1 className="text-red-500">{globalError}</h1>;
  }

  return (
    <div className="flex h-screen flex-col bg-gradient-to-br from-green-50 to-blue-50">
      {/* Game Header */}
      <div className="flex-shrink-0 border-b border-green-200 bg-white/80 shadow-sm backdrop-blur-sm">
        <div className="mx-auto max-w-7xl px-2 py-3 sm:px-4">
          <div className="flex items-center justify-between">
            {/* Left side - Game info */}
            <div className="flex items-center space-x-2 sm:space-x-6">
              <div className="flex items-center space-x-2">
                <Trophy className="h-4 w-4 text-green-600 sm:h-5 sm:w-5" />
                <h1 className="text-lg font-bold text-gray-800 sm:text-xl">
                  Taka Match
                </h1>
              </div>

              <div className="flex items-center space-x-2 text-xs text-gray-600 sm:text-sm">
                <Clock className="h-3 w-3 sm:h-4 sm:w-4" />
                <span>{formatTime(gameTime)}</span>
              </div>

              <div className="hidden items-center space-x-2 text-sm text-gray-600 sm:flex">
                <Users className="h-4 w-4" />
                <span>Online Match</span>
              </div>
            </div>

            {/* Right side - Game controls */}
            <div className="flex items-center space-x-2 sm:space-x-4">
              <div className="hidden text-xs text-gray-600 sm:text-sm md:block">
                Game ID: <span className="font-mono">{gameId || "Local"}</span>
              </div>

              <button className="flex items-center space-x-1 rounded-lg border border-gray-300 bg-white px-2 py-1.5 text-xs text-gray-600 hover:bg-gray-50 sm:px-3 sm:text-sm">
                <Settings className="h-3 w-3 sm:h-4 sm:w-4" />
                <span className="hidden sm:inline">Settings</span>
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Main Game Area */}
      <div className="flex flex-1 flex-col p-2 sm:p-4 lg:flex-row">
        <div className="mx-auto flex h-full w-full max-w-7xl flex-col lg:flex-row">
          {/* Mobile/Tablet - Player Info Row */}
          <div className="mb-4 flex flex-col space-y-4 lg:hidden">
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              {/* Your Info */}
              <div className="rounded-lg border border-gray-200 bg-white p-3 shadow-sm">
                <div className="mb-2 flex items-center justify-between">
                  <h3 className="font-semibold text-gray-800">You</h3>
                  <div
                    className={`h-3 w-3 rounded-full ${playerColor === "white" ? "border-2 border-gray-400 bg-white" : "bg-gray-900"}`}
                  />
                </div>
                <div className="space-y-1 text-xs text-gray-600">
                  <div className="flex justify-between">
                    <span>Color:</span>
                    <span className="font-medium capitalize">
                      {playerColor}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span>Status:</span>
                    <span
                      className={`font-medium ${playerTurn === playerColor ? "text-green-600" : "text-gray-500"}`}
                    >
                      {playerTurn === playerColor ? "Your Turn" : "Waiting..."}
                    </span>
                  </div>
                </div>
              </div>

              {/* Opponent Info */}
              <div className="rounded-lg border border-gray-200 bg-white p-3 shadow-sm">
                <div className="mb-2 flex items-center justify-between">
                  <h3 className="font-semibold text-gray-800">Opponent</h3>
                  <div
                    className={`h-3 w-3 rounded-full ${playerColor === "black" ? "border-2 border-gray-400 bg-white" : "bg-gray-900"}`}
                  />
                </div>
                <div className="space-y-1 text-xs text-gray-600">
                  <div className="flex justify-between">
                    <span>Color:</span>
                    <span className="font-medium capitalize">
                      {playerColor === "white" ? "black" : "white"}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span>Status:</span>
                    <span
                      className={`font-medium ${playerTurn !== playerColor ? "text-green-600" : "text-gray-500"}`}
                    >
                      {playerTurn !== playerColor ? "Their Turn" : "Waiting..."}
                    </span>
                  </div>
                </div>
              </div>
            </div>

            {/* Mobile Actions */}
            <div className="flex gap-2 overflow-x-auto">
              <button className="rounded-md bg-green-600 px-3 py-2 text-xs font-medium whitespace-nowrap text-white hover:bg-green-700 disabled:opacity-50">
                End Turn
              </button>
              <button className="rounded-md border border-gray-300 bg-white px-3 py-2 text-xs font-medium whitespace-nowrap text-gray-700 hover:bg-gray-50">
                Request Draw
              </button>
              <button className="rounded-md border border-red-300 bg-white px-3 py-2 text-xs font-medium whitespace-nowrap text-red-600 hover:bg-red-50">
                Forfeit
              </button>
            </div>
          </div>

          {/* Desktop Left Sidebar - Player 1 Info */}
          <div className="hidden w-64 flex-col space-y-4 pr-4 lg:flex">
            <div className="rounded-lg border border-gray-200 bg-white p-4 shadow-sm">
              <div className="mb-3 flex items-center justify-between">
                <h3 className="font-semibold text-gray-800">You</h3>
                <div
                  className={`h-4 w-4 rounded-full ${playerColor === "white" ? "border-2 border-gray-400 bg-white" : "bg-gray-900"}`}
                />
              </div>
              <div className="space-y-2 text-sm text-gray-600">
                <div className="flex justify-between">
                  <span>Color:</span>
                  <span className="font-medium capitalize">{playerColor}</span>
                </div>
                <div className="flex justify-between">
                  <span>Score:</span>
                  <span className="font-medium">0</span>
                </div>
                <div className="flex justify-between">
                  <span>Status:</span>
                  <span
                    className={`font-medium ${playerTurn === playerColor ? "text-green-600" : "text-gray-500"}`}
                  >
                    {playerTurn === playerColor ? "Your Turn" : "Waiting..."}
                  </span>
                </div>
              </div>

              {playerTurn === playerColor && (
                <div className="mt-4 rounded-md border border-green-200 bg-green-50 p-3">
                  <p className="text-sm font-medium text-green-800">
                    It&#39;s your turn!
                  </p>
                  <p className="mt-1 text-xs text-green-600">
                    Select a piece to make your move
                  </p>
                </div>
              )}
            </div>

            {/* Desktop Game Actions */}
            <div className="rounded-lg border border-gray-200 bg-white p-4 shadow-sm">
              <h3 className="mb-3 font-semibold text-gray-800">Actions</h3>
              <div className="space-y-2">
                <button className="w-full rounded-md bg-green-600 px-3 py-2 text-sm font-medium text-white hover:bg-green-700 disabled:opacity-50">
                  End Turn
                </button>
                <button className="w-full rounded-md border border-gray-300 bg-white px-3 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50">
                  Request Draw
                </button>
                <button className="w-full rounded-md border border-red-300 bg-white px-3 py-2 text-sm font-medium text-red-600 hover:bg-red-50">
                  Forfeit
                </button>
              </div>
            </div>
          </div>

          {/* Center - Game Board */}
          <div className="min-h-0 flex-1">
            <GameBoard />
          </div>

          {/* Desktop Right Sidebar - Player 2 Info */}
          <div className="hidden w-64 flex-col space-y-4 pl-4 lg:flex">
            <div className="rounded-lg border border-gray-200 bg-white p-4 shadow-sm">
              <div className="mb-3 flex items-center justify-between">
                <h3 className="font-semibold text-gray-800">Opponent</h3>
                <div
                  className={`h-4 w-4 rounded-full ${playerColor === "black" ? "border-2 border-gray-400 bg-white" : "bg-gray-900"}`}
                />
              </div>
              <div className="space-y-2 text-sm text-gray-600">
                <div className="flex justify-between">
                  <span>Color:</span>
                  <span className="font-medium capitalize">
                    {playerColor === "white" ? "black" : "white"}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Score:</span>
                  <span className="font-medium">0</span>
                </div>
                <div className="flex justify-between">
                  <span>Status:</span>
                  <span
                    className={`font-medium ${playerTurn !== playerColor ? "text-green-600" : "text-gray-500"}`}
                  >
                    {playerTurn !== playerColor ? "Their Turn" : "Waiting..."}
                  </span>
                </div>
              </div>

              {playerTurn !== playerColor && (
                <div className="mt-4 rounded-md border border-blue-200 bg-blue-50 p-3">
                  <p className="text-sm font-medium text-blue-800">
                    Opponent&#39;s turn
                  </p>
                  <p className="mt-1 text-xs text-blue-600">
                    Waiting for their move...
                  </p>
                </div>
              )}
            </div>

            {/* Move History */}
            <div className="rounded-lg border border-gray-200 bg-white p-4 shadow-sm">
              <h3 className="mb-3 font-semibold text-gray-800">Move History</h3>
              <div className="max-h-64 space-y-1 overflow-y-auto text-sm">
                <div className="flex justify-between text-gray-600">
                  <span>1. White:</span>
                  <span>Move W1 to (3,3)</span>
                </div>
                <div className="flex justify-between text-gray-600">
                  <span>1. Black:</span>
                  <span>Move B1 to (10,4)</span>
                </div>
                <div className="py-2 text-center text-xs text-gray-400">
                  Game started
                </div>
              </div>
            </div>
          </div>

          {/* Mobile Bottom Panel - Move History */}
          <div className="mt-4 rounded-lg border border-gray-200 bg-white p-3 shadow-sm lg:hidden">
            <h3 className="mb-2 font-semibold text-gray-800">Recent Moves</h3>
            <div className="max-h-32 space-y-1 overflow-y-auto text-xs">
              <div className="flex justify-between text-gray-600">
                <span>1. White:</span>
                <span>W1→(3,3)</span>
              </div>
              <div className="flex justify-between text-gray-600">
                <span>1. Black:</span>
                <span>B1→(10,4)</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PlayPage;
