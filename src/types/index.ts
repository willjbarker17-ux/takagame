import type { Request, Response, NextFunction } from "express";
import type { User } from "@prisma/client";

export type AuthenticatedRequest = Request & { user?: User };

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
