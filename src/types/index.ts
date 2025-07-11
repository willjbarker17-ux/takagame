import type { Request, Response, NextFunction } from "express";
import type { Payload } from "@/utils/tokens";

export type RequestWithUser = Request & { user?: Payload };

export type RouteHandler = (
  req: RequestWithUser,
  res: Response,
  next: NextFunction,
) => Promise<void> | void;

export type ErrorHandler = (
  err: Error,
  req: Request,
  res: Response,
  next: NextFunction,
) => Promise<void> | void;
