"use client";

import { io, Socket } from "socket.io-client";

// Type definitions from frontend.md
interface GamePiece {
  id: string;
  playerId: string;
  x?: number; // Optional for unactivated goalies
  y?: number; // Optional for unactivated goalies
  type?: string;
  hasBall?: boolean;
  facingDirection?: string;
}

interface BallPosition {
  x: number;
  y: number;
}

interface GameState {
  pieces: GamePiece[];
  ballPositions: BallPosition[];
}

interface Player {
  id: string;
  username: string;
  elo: number;
}

interface Game {
  id: string;

  // Registered user players (optional)
  whitePlayerId: string | null;
  blackPlayerId: string | null;

  // Guest player identifiers (alternative to registered users)
  whitePlayerGuestId: string | null;
  whitePlayerUsername: string | null;
  blackPlayerGuestId: string | null;
  blackPlayerUsername: string | null;

  status: "waiting" | "active" | "completed";
  currentTurn: "white" | "black" | null;
  gameState: GameState | null;
  ballPositions: BallPosition[] | null;
  winner: "white" | "black" | null;

  // These may be null for guest games
  whitePlayer: Player | null;
  blackPlayer: Player | null;
}

interface GuestSession {
  sessionId: string;
  username: string;
}

// Move data interface
interface MoveData {
  playerId: string;
  username: string;
  timestamp: string;
  type: string;
  details?: unknown;
}

// Auth config interface
interface AuthConfig {
  auth: {
    guestSessionId?: string;
    guestUsername?: string;
    token?: string;
    userId?: string;
  };
}

// Event callback types
type GameJoinedCallback = (data: { gameId: string; game: Game }) => void;
type PlayerJoinedCallback = (data: {
  playerId: string;
  username: string;
  game: Game;
}) => void;
type GameStateUpdatedCallback = (data: {
  game: Game;
  gameState: GameState;
  move: MoveData;
}) => void;
type MoveConfirmedCallback = (data: {
  game: Game;
  gameState: GameState;
}) => void;
type GameOverCallback = (data: {
  winner: string;
  winnerUsername: string;
  game: Game;
}) => void;
type ErrorCallback = (error: { message: string }) => void;
type ConnectionCallback = () => void;
type GuestSessionCreatedCallback = (session: GuestSession) => void;

// Authentication method types
type AuthMethod =
  | { accessToken: string }
  | { guestUsername: string }
  | { guestSession: GuestSession };

export class GameClient {
  private socket: Socket | null = null;
  private currentGame: Game | null = null;
  private isGuest: boolean;
  private guestSession?: GuestSession;
  private guestUsername?: string;
  private accessToken?: string;
  private isConnected: boolean = false;
  private isConnecting: boolean = false;

  // Event callbacks
  private onGameJoined: GameJoinedCallback | null = null;
  private onPlayerJoined: PlayerJoinedCallback | null = null;
  private onGameStateUpdated: GameStateUpdatedCallback | null = null;
  private onMoveConfirmed: MoveConfirmedCallback | null = null;
  private onGameOver: GameOverCallback | null = null;
  private onError: ErrorCallback | null = null;
  private onConnect: ConnectionCallback | null = null;
  private onDisconnect: ConnectionCallback | null = null;
  private onGuestSessionCreated: GuestSessionCreatedCallback | null = null;

  constructor(authMethod: AuthMethod) {
    if ("accessToken" in authMethod) {
      this.isGuest = false;
      this.accessToken = authMethod.accessToken;
    } else if ("guestUsername" in authMethod) {
      this.isGuest = true;
      this.guestUsername = authMethod.guestUsername;
    } else {
      this.isGuest = true;
      this.guestSession = authMethod.guestSession;
    }
  }

  // Event listener methods
  setOnGameJoined(callback: GameJoinedCallback) {
    this.onGameJoined = callback;
  }

  setOnPlayerJoined(callback: PlayerJoinedCallback) {
    this.onPlayerJoined = callback;
  }

  setOnGameStateUpdated(callback: GameStateUpdatedCallback) {
    this.onGameStateUpdated = callback;
  }

  setOnMoveConfirmed(callback: MoveConfirmedCallback) {
    this.onMoveConfirmed = callback;
  }

  setOnGameOver(callback: GameOverCallback) {
    this.onGameOver = callback;
  }

  setOnError(callback: ErrorCallback) {
    this.onError = callback;
  }

  setOnConnect(callback: ConnectionCallback) {
    this.onConnect = callback;
  }

  setOnDisconnect(callback: ConnectionCallback) {
    this.onDisconnect = callback;
  }

  setOnGuestSessionCreated(callback: GuestSessionCreatedCallback) {
    this.onGuestSessionCreated = callback;
  }

  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      if (this.socket?.connected) {
        resolve();
        return;
      }

      if (this.isConnecting) {
        const checkConnection = () => {
          if (this.isConnected) {
            resolve();
          } else if (!this.isConnecting) {
            reject(new Error("Connection attempt failed"));
          } else {
            setTimeout(checkConnection, 100);
          }
        };
        checkConnection();
        return;
      }

      this.isConnecting = true;

      // Setup auth configuration based on user type
      let authConfig: AuthConfig;
      if (this.isGuest) {
        if (this.guestSession) {
          authConfig = {
            auth: {
              guestSessionId: this.guestSession.sessionId,
              guestUsername: this.guestSession.username, // fallback
            },
          };
        } else if (this.guestUsername) {
          authConfig = {
            auth: {
              guestUsername: this.guestUsername,
            },
          };
        } else {
          // Fallback for new guest users
          const guestUsername = "Player" + Math.floor(Math.random() * 1000);
          authConfig = {
            auth: {
              guestUsername: guestUsername,
            },
          };
        }
      } else {
        authConfig = {
          auth: {
            token: this.accessToken,
          },
        };
      }

      const serverUrl =
        process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:3000";
      console.log(
        "Connecting to socket server:",
        serverUrl,
        "with auth:",
        authConfig.auth,
      );
      console.log(
        "Socket auth debug - isGuest:",
        this.isGuest,
        "guestSession:",
        this.guestSession,
        "guestUsername:",
        this.guestUsername,
      );

      this.socket = io(serverUrl, {
        ...authConfig,
        timeout: 10000,
        transports: ["websocket", "polling"], // Try websocket first, fallback to polling
      });

      // Connection events
      this.socket.on("connect", () => {
        console.log(
          "Connected to game server as",
          this.isGuest ? "guest" : "authenticated user",
        );
        this.isConnected = true;
        this.isConnecting = false;
        this.onConnect?.();
        resolve();
      });

      this.socket.on("disconnect", () => {
        console.log("Disconnected from game server");
        this.isConnected = false;
        this.onDisconnect?.();
      });

      this.socket.on("connect_error", (error) => {
        console.error("Connection failed:", error.message);
        this.isConnecting = false;
        this.isConnected = false;

        if (error.message.includes("Authentication required")) {
          this.onError?.({
            message: "Authentication failed. Please provide a username.",
          });
        } else if (error.message.includes("Invalid or expired guest session")) {
          this.onError?.({
            message: "Session expired. Please start a new game.",
          });
        } else if (error.message === "Access denied") {
          this.onError?.({
            message:
              "Access denied. You are not authorized to perform this action.",
          });
        } else {
          this.onError?.({ message: `Connection failed: ${error.message}` });
        }

        reject(error);
      });

      // Game events
      this.socket.on("game-joined", this.handleGameJoined.bind(this));
      this.socket.on("player-joined", this.handlePlayerJoined.bind(this));
      this.socket.on(
        "game-state-updated",
        this.handleGameStateUpdated.bind(this),
      );
      this.socket.on("move-confirmed", this.handleMoveConfirmed.bind(this));
      this.socket.on("game-over", this.handleGameOver.bind(this));
      this.socket.on("error", this.handleError.bind(this));

      // Guest session events
      this.socket.on(
        "guest-session-created",
        this.handleGuestSessionCreated.bind(this),
      );
    });
  }

  async createGame(guestUsername?: string): Promise<string> {
    try {
      if (this.isGuest) {
        const response = await fetch(
          `${process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:3000"}/games/create-guest`,
          {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({
              guestUsername: guestUsername,
            }),
          },
        );

        if (!response.ok) {
          throw new Error(
            `Failed to create guest game: ${response.statusText}`,
          );
        }

        const data = await response.json();
        this.guestSession = data.guestSession; // Store session for reconnections
        return data.gameId;
      } else {
        const response = await fetch(
          `${process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:3000"}/games/create`,
          {
            method: "POST",
            headers: {
              Authorization: `Bearer ${this.accessToken}`,
              "Content-Type": "application/json",
            },
            body: JSON.stringify({}),
          },
        );

        if (!response.ok) {
          throw new Error(`Failed to create game: ${response.statusText}`);
        }

        const data = await response.json();
        return data.gameId;
      }
    } catch (error) {
      console.error("Error creating game:", error);
      throw error;
    }
  }

  async getGame(gameId: string): Promise<Game> {
    try {
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:3000"}/games/${gameId}`,
      );

      if (!response.ok) {
        throw new Error(`Failed to get game: ${response.statusText}`);
      }

      const data = await response.json();
      return data.game;
    } catch (error) {
      console.error("Error getting game:", error);
      throw error;
    }
  }

  joinGame(gameId: string): void {
    if (!this.socket?.connected) {
      throw new Error("Socket not connected. Call connect() first.");
    }

    this.socket.emit("join-game", gameId);
  }

  makeMove(gameState: GameState): void {
    if (!this.currentGame) {
      throw new Error("No active game to make move in.");
    }

    if (!this.socket?.connected) {
      throw new Error("Socket not connected.");
    }

    this.socket.emit("make-move", {
      gameId: this.currentGame.id,
      gameState: gameState,
    });
  }

  // Event handlers
  private handleGameJoined(data: { gameId: string; game: Game }): void {
    this.currentGame = data.game;
    console.log("Joined game:", data.gameId);
    this.onGameJoined?.(data);
  }

  private handlePlayerJoined(data: {
    playerId: string;
    username: string;
    game: Game;
  }): void {
    this.currentGame = data.game;
    console.log("Player joined:", data.username);
    this.onPlayerJoined?.(data);
  }

  private handleGameStateUpdated(data: {
    game: Game;
    gameState: GameState;
    move: MoveData;
  }): void {
    this.currentGame = data.game;
    console.log("Game state updated by:", data.move.username);
    this.onGameStateUpdated?.(data);
  }

  private handleMoveConfirmed(data: {
    game: Game;
    gameState: GameState;
  }): void {
    this.currentGame = data.game;
    console.log("Move confirmed");
    this.onMoveConfirmed?.(data);
  }

  private handleGameOver(data: {
    winner: string;
    winnerUsername: string;
    game: Game;
  }): void {
    this.currentGame = data.game;
    console.log("Game over! Winner:", data.winnerUsername);
    this.onGameOver?.(data);
  }

  private handleError(error: { message: string }): void {
    console.error("Game error:", error.message);

    // Treat "Access denied" as a connection error
    if (error.message === "Access denied") {
      this.onError?.({
        message:
          "Access denied. You are not authorized to perform this action.",
      });
    } else {
      this.onError?.(error);
    }
  }

  private handleGuestSessionCreated(sessionData: GuestSession): void {
    console.log("Guest session created by server:", sessionData);
    this.guestSession = sessionData;
    this.onGuestSessionCreated?.(sessionData);
  }

  // Utility methods
  getCurrentGame(): Game | null {
    return this.currentGame;
  }

  isSocketConnected(): boolean {
    return this.isConnected;
  }

  isGuestUser(): boolean {
    return this.isGuest;
  }

  getGuestSession(): GuestSession | undefined {
    return this.guestSession;
  }

  getGuestUsername(): string | undefined {
    return this.guestUsername;
  }

  disconnect(): void {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }
    this.isConnected = false;
    this.isConnecting = false;
    this.currentGame = null;
  }

  // Update access token (useful for token refresh)
  updateAccessToken(newToken: string): void {
    if (!this.isGuest) {
      this.accessToken = newToken;
      if (this.socket) {
        this.socket.auth = { token: newToken };
      }
    }
  }
}

// Singleton instance for global access
let gameClientInstance: GameClient | null = null;

export const initializeGameClient = (authMethod: AuthMethod): GameClient => {
  if (gameClientInstance) {
    gameClientInstance.disconnect();
  }
  gameClientInstance = new GameClient(authMethod);
  return gameClientInstance;
};

export const getGameClient = (): GameClient | null => {
  return gameClientInstance;
};

export const destroyGameClient = (): void => {
  if (gameClientInstance) {
    gameClientInstance.disconnect();
    gameClientInstance = null;
  }
};

// Export types for use in other components
export type {
  Game,
  GameState,
  GamePiece,
  BallPosition,
  GuestSession,
  AuthMethod,
};
