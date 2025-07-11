import { describe, it, expect, vi, beforeEach } from "vitest";
import mockDatabase from "@tests/mocks/database";

vi.mock("@/database", () => ({ default: mockDatabase }));

import type { Response, NextFunction } from "express";
import requireUser from "@/middleware/requireUser";
import { verifyToken } from "@/utils/tokens";
import type { AuthenticatedRequest } from "@/types";

vi.mock("@/utils/tokens", () => ({
  verifyToken: vi.fn(),
}));

describe("requireUser middleware", () => {
  let req: Partial<AuthenticatedRequest>;
  let res: Partial<Response>;
  let next: NextFunction;

  beforeEach(() => {
    req = {
      headers: {},
    };
    res = {
      sendStatus: vi.fn(),
    };
    next = vi.fn();
    vi.clearAllMocks();
  });

  it("should call next() and attach user profile if token is valid and user exists", async () => {
    const userPayload = { userId: "test-user-id", iat: 123, exp: 456 };
    const userProfile = {
      id: "test-user-id",
      createdAt: new Date(),
      updatedAt: new Date(),
      email: "test@example.com",
      username: "testuser",
      verified: true,
      elo: 500,
      onboardingComplete: true,
    };

    req.headers = { authorization: "Bearer valid.token.here" };
    vi.mocked(verifyToken).mockReturnValue(userPayload);
    mockDatabase.user.findUnique.mockResolvedValue(userProfile);

    await requireUser(req as AuthenticatedRequest, res as Response, next);

    expect(verifyToken).toHaveBeenCalledWith("valid.token.here");
    expect(mockDatabase.user.findUnique).toHaveBeenCalledWith({
      where: { id: "test-user-id" },
    });
    expect(req.user).toEqual(userProfile);
    expect(next).toHaveBeenCalledTimes(1);
    expect(res.sendStatus).not.toHaveBeenCalled();
  });

  it("should return 403 if token is valid but user doesn't exist in database", async () => {
    const userPayload = { userId: "test-user-id", iat: 123, exp: 456 };

    req.headers = { authorization: "Bearer valid.token.here" };
    vi.mocked(verifyToken).mockReturnValue(userPayload);
    mockDatabase.user.findUnique.mockResolvedValue(null);

    await requireUser(req as AuthenticatedRequest, res as Response, next);

    expect(verifyToken).toHaveBeenCalledWith("valid.token.here");
    expect(mockDatabase.user.findUnique).toHaveBeenCalledWith({
      where: { id: "test-user-id" },
    });
    expect(res.sendStatus).toHaveBeenCalledWith(403);
    expect(next).not.toHaveBeenCalled();
  });

  it("should call next with error if database query fails", async () => {
    const userPayload = { userId: "test-user-id", iat: 123, exp: 456 };
    const error = new Error("Database error");

    req.headers = { authorization: "Bearer valid.token.here" };
    vi.mocked(verifyToken).mockReturnValue(userPayload);
    mockDatabase.user.findUnique.mockRejectedValue(error);

    await requireUser(req as AuthenticatedRequest, res as Response, next);

    expect(verifyToken).toHaveBeenCalledWith("valid.token.here");
    expect(mockDatabase.user.findUnique).toHaveBeenCalledWith({
      where: { id: "test-user-id" },
    });
    expect(next).toHaveBeenCalledWith(error);
    expect(res.sendStatus).not.toHaveBeenCalled();
  });

  it("should return 401 if authorization header is missing", async () => {
    await requireUser(req as AuthenticatedRequest, res as Response, next);

    expect(res.sendStatus).toHaveBeenCalledWith(401);
    expect(next).not.toHaveBeenCalled();
    expect(verifyToken).not.toHaveBeenCalled();
    expect(mockDatabase.user.findUnique).not.toHaveBeenCalled();
  });

  it("should return 401 if authorization header does not start with 'Bearer '", async () => {
    req.headers = { authorization: "Invalid valid.token.here" };

    await requireUser(req as AuthenticatedRequest, res as Response, next);

    expect(res.sendStatus).toHaveBeenCalledWith(401);
    expect(next).not.toHaveBeenCalled();
    expect(mockDatabase.user.findUnique).not.toHaveBeenCalled();
  });

  it("should return 401 if token is missing from header", async () => {
    req.headers = { authorization: "Bearer " };

    await requireUser(req as AuthenticatedRequest, res as Response, next);

    expect(res.sendStatus).toHaveBeenCalledWith(401);
    expect(next).not.toHaveBeenCalled();
    expect(mockDatabase.user.findUnique).not.toHaveBeenCalled();
  });

  it("should return 403 if token is invalid", async () => {
    req.headers = { authorization: "Bearer invalid.token.here" };
    vi.mocked(verifyToken).mockReturnValue(null);

    await requireUser(req as AuthenticatedRequest, res as Response, next);

    expect(res.sendStatus).toHaveBeenCalledWith(403);
    expect(next).not.toHaveBeenCalled();
    expect(mockDatabase.user.findUnique).not.toHaveBeenCalled();
  });
});
