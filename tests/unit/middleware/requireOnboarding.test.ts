import { describe, it, expect, vi, beforeEach } from "vitest";
import type { Request, Response, NextFunction } from "express";
import requireOnboarding from "@/middleware/requireOnboarding";
import type { AuthenticatedRequest } from "@/types";
import mockDatabase from "@tests/mocks/database";

vi.mock("@/database", () => ({ default: mockDatabase }));

describe("requireOnboarding middleware", () => {
  let req: Partial<AuthenticatedRequest>;
  let res: Partial<Response>;
  let next: NextFunction;

  const mockUser = {
    id: "test-user-id",
    email: "test@example.com",
    username: "testuser",
    verified: true,
    elo: 1200,
    createdAt: new Date(),
    updatedAt: new Date(),
    onboardingComplete: true,
  };

  beforeEach(() => {
    req = {
      user: mockUser,
    };
    res = {
      status: vi.fn().mockReturnThis(),
      json: vi.fn(),
    };
    next = vi.fn();
    vi.clearAllMocks();
  });

  it("should call next() if user has completed onboarding", async () => {
    await requireOnboarding(req as Request, res as Response, next);

    expect(next).toHaveBeenCalledTimes(1);
    expect(next).toHaveBeenCalledWith();
    expect(res.status).not.toHaveBeenCalled();
    expect(res.json).not.toHaveBeenCalled();
  });

  it("should return 403 if user has not completed onboarding", async () => {
    req.user = { ...mockUser, onboardingComplete: false };

    await requireOnboarding(req as Request, res as Response, next);

    expect(res.status).toHaveBeenCalledWith(403);
    expect(res.json).toHaveBeenCalledWith({
      error: "Onboarding must be completed before accessing this resource",
      needsOnboarding: true,
    });
    expect(next).not.toHaveBeenCalled();
  });

  it("should throw error if req.user is missing (middleware used incorrectly)", () => {
    req.user = undefined;

    expect(() =>
      requireOnboarding(req as Request, res as Response, next),
    ).toThrow("req.user should exist, but doesn't");

    expect(res.status).not.toHaveBeenCalled();
    expect(res.json).not.toHaveBeenCalled();
    expect(next).not.toHaveBeenCalled();
  });
});
