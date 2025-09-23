"use client";

import React, { useEffect, useState, useRef } from "react";
import FullPageLoader from "@/components/FullPageLoader";
import GameBoard from "@/components/game/GameBoard";
import {
  useGameBoard,
  initializeSocketClient,
  connectToGame,
  createMultiplayerGame,
  disconnectFromGame,
} from "@/hooks/useGameStore";
import { useAuth } from "@/hooks/useAuth";
import { initializeGameClient } from "@/services/socketService";
import { Clock, Users, Trophy, Wifi, WifiOff, UserPlus } from "lucide-react";

const PlayPage: React.FC = () => {
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [gameTime, setGameTime] = useState<number>(0);
  const initializedRef = useRef(false);

  const {
    playerColor,
    playerTurn,
    gameId,
    isConnected,
    isConnecting,
    connectionError,
    gameStatus,
    whitePlayer,
    blackPlayer,
    waitingForOpponent,
    winner,
  } = useGameBoard();

  const auth = useAuth();

  // Debug logging for winner state
  React.useEffect(() => {
    console.log("Game state debug:", { winner, gameStatus, playerColor });
  }, [winner, gameStatus, playerColor]);

  // Guest authentication and socket initialization
  useEffect(() => {
    if (initializedRef.current || auth.isLoading) {
      return;
    }

    const initialize = async () => {
      initializedRef.current = true;
      setIsLoading(true);

      try {
        // If there's no valid auth (neither user nor guest), create a guest user.
        if (!auth.isAuthenticated && !auth.isGuest) {
          console.log("No valid auth, creating new guest user.");
          auth.loginAsGuest();
          // The effect will re-run once auth state is updated.
          initializedRef.current = false;
          return;
        }

        console.log("Auth state is ready:", {
          isGuest: auth.isGuest,
          guestSession: auth.guestSession,
          guestUsername: auth.guestUsername,
        });

        // At this point, we have some form of auth. Initialize the socket.
        let client;
        if (auth.guestSession) {
          client = initializeGameClient({ guestSession: auth.guestSession });
        } else if (auth.guestUsername) {
          client = initializeGameClient({ guestUsername: auth.guestUsername });
        } else {
          // Fallback for new guest users if somehow no username is set yet
          client = initializeGameClient({
            guestUsername: `Player${Math.floor(Math.random() * 10000)}`,
          });
        }

        client.setOnGuestSessionCreated((session) => {
          console.log("Received guest session from server:", session);
          if (
            !auth.guestSession ||
            auth.guestSession.sessionId !== session.sessionId
          ) {
            auth.setGuestSession(session);
          }
        });

        initializeSocketClient(client);

        const url = new URL(window.location.href);
        const gameIdFromUrl = url.searchParams.get("id");

        if (gameIdFromUrl) {
          await connectToGame(gameIdFromUrl);
        } else {
          const newGameId = await createMultiplayerGame(auth.guestUsername!);
          const newUrl = new URL(window.location.href);
          newUrl.searchParams.set("id", newGameId);
          window.history.replaceState({}, "", newUrl.toString());
          await connectToGame(newGameId);
        }
      } catch (error) {
        console.error("Failed to initialize game:", error);
        initializedRef.current = false; // Allow re-initialization on error
      } finally {
        setIsLoading(false);
      }
    };

    initialize();
  }, [auth, auth.isLoading, auth.loginAsGuest, auth.setGuestSession]); // Only depend on stable values

  // Cleanup effect - separate from initialization
  useEffect(() => {
    return () => {
      if (initializedRef.current) {
        disconnectFromGame();
      }
    };
  }, []);

  // Game timer effect
  useEffect(() => {
    let timer: NodeJS.Timeout;

    if (gameStatus === "active") {
      timer = setInterval(() => {
        setGameTime((prev) => prev + 1);
      }, 1000);
    }

    return () => {
      if (timer) clearInterval(timer);
    };
  }, [gameStatus]);

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
  };

  const getConnectionStatusIcon = () => {
    if (isConnecting)
      return <Clock className="h-4 w-4 animate-spin text-yellow-600" />;
    if (isConnected) return <Wifi className="h-4 w-4 text-green-600" />;
    return <WifiOff className="h-4 w-4 text-red-600" />;
  };

  const getConnectionStatusText = () => {
    if (isConnecting) return "Connecting...";
    if (isConnected) return "Connected";
    if (connectionError) return `Error: ${connectionError}`;
    return "Disconnected";
  };

  const getOpponentInfo = () => {
    const opponent = playerColor === "white" ? blackPlayer : whitePlayer;
    return {
      username: opponent?.username || "Waiting...",
      elo: opponent?.elo || 0,
      color: playerColor === "white" ? "black" : "white",
    };
  };

  const getCurrentPlayerInfo = () => {
    const currentPlayer = playerColor === "white" ? whitePlayer : blackPlayer;
    return {
      username: currentPlayer?.username || "Player",
      elo: currentPlayer?.elo || 0,
      color: playerColor,
    };
  };

  if (isLoading || isConnecting) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-gradient-to-br from-green-50 to-blue-50">
        <div className="text-center">
          <FullPageLoader />
          <p className="mt-4 text-gray-600">
            {isConnecting ? "Connecting to game..." : "Setting up your game..."}
          </p>
        </div>
      </div>
    );
  }

  if (connectionError && !isConnected) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-gradient-to-br from-green-50 to-blue-50">
        <div className="max-w-md rounded-lg border border-red-200 bg-white p-6 text-center shadow-lg">
          <WifiOff className="mx-auto mb-4 h-12 w-12 text-red-600" />
          <h1 className="mb-2 text-lg font-semibold text-red-800">
            Connection Error
          </h1>
          <p className="mb-4 text-red-600">{connectionError}</p>
          <button
            onClick={() => window.location.reload()}
            className="rounded-md bg-red-600 px-4 py-2 text-white hover:bg-red-700"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  const currentPlayerInfo = getCurrentPlayerInfo();
  const opponentInfo = getOpponentInfo();

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

              <div className="flex items-center space-x-2 text-xs text-gray-600 sm:text-sm">
                {getConnectionStatusIcon()}
                <span className="hidden sm:inline">
                  {getConnectionStatusText()}
                </span>
              </div>

              <div className="hidden items-center space-x-2 text-sm text-gray-600 sm:flex">
                {waitingForOpponent ? (
                  <>
                    <UserPlus className="h-4 w-4 text-orange-600" />
                    <span className="text-orange-600">
                      Waiting for opponent...
                    </span>
                  </>
                ) : (
                  <>
                    <Users className="h-4 w-4" />
                    <span>Online Match</span>
                  </>
                )}
              </div>
            </div>

            {/* Right side - Game controls */}
            <div className="flex items-center space-x-2 sm:space-x-4">
              <div className="hidden text-xs text-gray-600 sm:text-sm md:block">
                Game ID: <span className="font-mono">{gameId || "Local"}</span>
              </div>
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
                  <h3 className="font-semibold text-gray-800">
                    {currentPlayerInfo.username}
                  </h3>
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
                    <span>ELO:</span>
                    <span className="font-medium">{currentPlayerInfo.elo}</span>
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
                  <h3 className="font-semibold text-gray-800">
                    {waitingForOpponent ? "Waiting..." : opponentInfo.username}
                  </h3>
                  <div
                    className={`h-3 w-3 rounded-full ${opponentInfo.color === "white" ? "border-2 border-gray-400 bg-white" : "bg-gray-900"}`}
                  />
                </div>
                <div className="space-y-1 text-xs text-gray-600">
                  <div className="flex justify-between">
                    <span>Color:</span>
                    <span className="font-medium capitalize">
                      {opponentInfo.color}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span>ELO:</span>
                    <span className="font-medium">
                      {waitingForOpponent ? "-" : opponentInfo.elo}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span>Status:</span>
                    <span
                      className={`font-medium ${
                        waitingForOpponent
                          ? "text-orange-600"
                          : playerTurn !== playerColor
                            ? "text-green-600"
                            : "text-gray-500"
                      }`}
                    >
                      {waitingForOpponent
                        ? "Joining..."
                        : playerTurn !== playerColor
                          ? "Their Turn"
                          : "Waiting..."}
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
                <h3 className="font-semibold text-gray-800">
                  {currentPlayerInfo.username}
                </h3>
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
                  <span>ELO:</span>
                  <span className="font-medium">{currentPlayerInfo.elo}</span>
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

              {playerTurn === playerColor && !waitingForOpponent && (
                <div className="mt-4 rounded-md border border-green-200 bg-green-50 p-3">
                  <p className="text-sm font-medium text-green-800">
                    It&#39;s your turn!
                  </p>
                  <p className="mt-1 text-xs text-green-600">
                    Select a piece to make your move
                  </p>
                </div>
              )}

              {waitingForOpponent && (
                <div className="mt-4 rounded-md border border-orange-200 bg-orange-50 p-3">
                  <p className="text-sm font-medium text-orange-800">
                    Waiting for opponent
                  </p>
                  <p className="mt-1 text-xs text-orange-600">
                    Share the game URL to invite someone
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
                <h3 className="font-semibold text-gray-800">
                  {waitingForOpponent ? "Waiting..." : opponentInfo.username}
                </h3>
                <div
                  className={`h-4 w-4 rounded-full ${opponentInfo.color === "white" ? "border-2 border-gray-400 bg-white" : "bg-gray-900"}`}
                />
              </div>
              <div className="space-y-2 text-sm text-gray-600">
                <div className="flex justify-between">
                  <span>Color:</span>
                  <span className="font-medium capitalize">
                    {opponentInfo.color}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>ELO:</span>
                  <span className="font-medium">
                    {waitingForOpponent ? "-" : opponentInfo.elo}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Status:</span>
                  <span
                    className={`font-medium ${
                      waitingForOpponent
                        ? "text-orange-600"
                        : playerTurn !== playerColor
                          ? "text-green-600"
                          : "text-gray-500"
                    }`}
                  >
                    {waitingForOpponent
                      ? "Joining..."
                      : playerTurn !== playerColor
                        ? "Their Turn"
                        : "Waiting..."}
                  </span>
                </div>
              </div>

              {playerTurn !== playerColor && !waitingForOpponent && (
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

      {/* Winning Animation Overlay */}
      {winner && gameStatus === "completed" && (
        <div className="bg-opacity-50 fixed inset-0 z-50 flex items-center justify-center bg-black backdrop-blur-sm">
          <div className="animate-bounce rounded-lg bg-white p-8 text-center shadow-2xl">
            <div className="mb-4">
              {winner === playerColor ? (
                <>
                  <div className="mb-2 text-6xl">🎉</div>
                  <h2 className="mb-2 text-3xl font-bold text-green-600">
                    You Won!
                  </h2>
                  <p className="text-lg text-gray-700">
                    Congratulations! You scored a goal and won the match!
                  </p>
                </>
              ) : (
                <>
                  <div className="mb-2 text-6xl">😔</div>
                  <h2 className="mb-2 text-3xl font-bold text-red-600">
                    You Lost!
                  </h2>
                  <p className="text-lg text-gray-700">
                    Better luck next time! Your opponent scored the winning
                    goal.
                  </p>
                </>
              )}
            </div>
            <div className="space-x-4">
              <button
                onClick={() => (window.location.href = "/")}
                className="rounded-md bg-blue-600 px-6 py-2 text-white hover:bg-blue-700"
              >
                New Game
              </button>
              <button
                onClick={() => window.location.reload()}
                className="rounded-md border border-gray-300 bg-white px-6 py-2 text-gray-700 hover:bg-gray-50"
              >
                Play Again
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default PlayPage;
