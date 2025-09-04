import type { Request, Response, NextFunction } from "express";
import type { Socket } from "socket.io";
import type { User } from "@prisma/client";
import type { GuestSession } from "@/utils/guestSession";

export type AuthenticatedRequest = Request & { user?: User };

export type AuthenticatedSocket = Socket & {
  user?: User;
  guestUser?: GuestSession;
};

export type AuthenticatedRouteHandler = (
  req: AuthenticatedRequest,
  res: Response,
  next: NextFunction,
) => Promise<void> | void;

export type RouteHandler = (
  req: Request,
  res: Response,
  next: NextFunction,
) => Promise<void> | void;

export type ErrorHandler = (
  err: Error,
  req: Request,
  res: Response,
  next: NextFunction,
) => Promise<void> | void;

export interface GamePiece {
  id: string;
  playerId: string;
  x?: number; // Optional for unactivated goalies
  y?: number; // Optional for unactivated goalies
  type?: string;
  hasBall?: boolean;
  facingDirection?: string;
}

export interface BallPosition {
  x: number;
  y: number;
}

export interface GameState {
  pieces: GamePiece[];
  ballPositions: BallPosition[];
}
