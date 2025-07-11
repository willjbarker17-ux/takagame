import { describe, it, expect, vi, beforeEach } from "vitest";
import type { Request, Response, NextFunction } from "express";
import requireOnboarding from "@/middleware/requireOnboarding";
import prisma from "@tests/mocks/database";
import type { RequestWithUser } from "@/types";

describe("requireOnboarding middleware", () => {
  let req: Partial<RequestWithUser>;
  let res: Partial<Response>;
  let next: NextFunction;

  beforeEach(() => {
    req = {
      user: { userId: "test-user-id" },
    };
    res = {
      status: vi.fn().mockReturnThis(),
      json: vi.fn(),
    };
    next = vi.fn();
    vi.clearAllMocks();
  });

  it("should call next() if user has completed onboarding", async () => {
    prisma.user.findUnique.mockResolvedValue({
      id: "test-user-id",
      email: "test@example.com",
      username: "testuser",
      onboardingComplete: true,
      createdAt: new Date(),
      updatedAt: new Date(),
      verified: true,
      elo: 1200,
    });

    await requireOnboarding(req as Request, res as Response, next);

    expect(prisma.user.findUnique).toHaveBeenCalledWith({
      where: { id: "test-user-id" },
      select: { onboardingComplete: true },
    });
    expect(next).toHaveBeenCalledTimes(1);
    expect(next).toHaveBeenCalledWith();
    expect(res.status).not.toHaveBeenCalled();
    expect(res.json).not.toHaveBeenCalled();
  });

  it("should return 403 if user has not completed onboarding", async () => {
    prisma.user.findUnique.mockResolvedValue({
      id: "test-user-id",
      email: "test@example.com",
      username: "testuser",
      onboardingComplete: false,
      createdAt: new Date(),
      updatedAt: new Date(),
      verified: true,
      elo: 1200,
    });

    await requireOnboarding(req as Request, res as Response, next);

    expect(res.status).toHaveBeenCalledWith(403);
    expect(res.json).toHaveBeenCalledWith({
      error: "Onboarding must be completed before accessing this resource",
      needsOnboarding: true,
    });
    expect(next).not.toHaveBeenCalled();
  });

  it("should return 404 if user from token is not found in database", async () => {
    prisma.user.findUnique.mockResolvedValue(null);

    await requireOnboarding(req as Request, res as Response, next);

    expect(res.status).toHaveBeenCalledWith(404);
    expect(res.json).toHaveBeenCalledWith({ error: "User not found" });
    expect(next).not.toHaveBeenCalled();
  });

  it("should return 401 if req.user is missing", async () => {
    req.user = undefined;

    await requireOnboarding(req as Request, res as Response, next);

    expect(res.status).toHaveBeenCalledWith(401);
    expect(res.json).toHaveBeenCalledWith({ error: "Authentication required" });
    expect(next).not.toHaveBeenCalled();
    expect(prisma.user.findUnique).not.toHaveBeenCalled();
  });

  it("should pass error to next() if prisma throws an error", async () => {
    const dbError = new Error("Database connection failed");
    prisma.user.findUnique.mockRejectedValue(dbError);

    await requireOnboarding(req as Request, res as Response, next);

    expect(next).toHaveBeenCalledTimes(1);
    expect(next).toHaveBeenCalledWith(dbError);
    expect(res.status).not.toHaveBeenCalled();
    expect(res.json).not.toHaveBeenCalled();
  });
});
