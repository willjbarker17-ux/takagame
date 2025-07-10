import type { Request, Response, NextFunction } from "express";
import type { Payload } from "@/utils/tokens";

export type RouteHandler = (
  req: Request & { user?: Payload },
  res: Response,
  next: NextFunction,
) => Promise<void> | void;

export type ErrorHandler = (
  err: Error,
  req: Request,
  res: Response,
  next: NextFunction,
) => Promise<void> | void;
