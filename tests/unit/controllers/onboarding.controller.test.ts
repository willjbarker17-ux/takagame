import { describe, it, expect, vi, beforeEach } from "vitest";
import mockDatabase from "@tests/mocks/database";

vi.mock("@/database", () => ({ default: mockDatabase }));

import type { Response, NextFunction } from "express";
import { completeOnboarding } from "@/controllers/onboarding.controller";
import type { AuthenticatedRequest } from "@/types";

describe("onboarding.controller: completeOnboarding", () => {
  let req: Partial<AuthenticatedRequest>;
  let res: Partial<Response>;
  let next: NextFunction;

  const userId = "user_123";
  const validUsername = "testuser123";
  const validSkillLevel = "intermediate";

  const mockUser = {
    id: userId,
    email: "test@example.com",
    username: "temp_username",
    verified: true,
    onboardingComplete: false,
    elo: 200,
    createdAt: new Date(),
    updatedAt: new Date(),
  };

  beforeEach(() => {
    req = {
      user: mockUser,
      body: {
        username: validUsername,
        skillLevel: validSkillLevel,
      },
    };
    res = {
      status: vi.fn().mockReturnThis(),
      json: vi.fn().mockReturnThis(),
    };
    next = vi.fn();

    vi.clearAllMocks();
  });

  it("should successfully complete onboarding for valid user", async () => {
    const updatedUser = {
      ...mockUser,
      username: validUsername,
      elo: 500, // intermediate level
      onboardingComplete: true,
    };

    mockDatabase.user.findUnique.mockResolvedValue(null); // username check
    mockDatabase.user.update.mockResolvedValue(updatedUser);

    await completeOnboarding(
      req as AuthenticatedRequest,
      res as Response,
      next,
    );

    expect(mockDatabase.user.findUnique).toHaveBeenCalledWith({
      where: { username: validUsername },
    });

    expect(mockDatabase.user.update).toHaveBeenCalledWith({
      where: { id: userId },
      data: {
        username: validUsername,
        elo: 500,
        onboardingComplete: true,
      },
    });

    expect(res.json).toHaveBeenCalledWith(updatedUser);
    expect(next).not.toHaveBeenCalled();
  });

  it("should map skill levels to correct ELO ratings", async () => {
    const testCases = [
      { skillLevel: "beginner", expectedElo: 200 },
      { skillLevel: "intermediate", expectedElo: 500 },
      { skillLevel: "advanced", expectedElo: 1000 },
    ];

    for (const { skillLevel, expectedElo } of testCases) {
      vi.clearAllMocks();

      req.body = { username: validUsername, skillLevel };

      mockDatabase.user.findUnique.mockResolvedValueOnce(null);
      mockDatabase.user.update.mockResolvedValue({
        ...mockUser,
        elo: expectedElo,
        onboardingComplete: true,
      });

      await completeOnboarding(
        req as AuthenticatedRequest,
        res as Response,
        next,
      );

      expect(mockDatabase.user.update).toHaveBeenCalledWith({
        where: { id: userId },
        data: {
          username: validUsername,
          elo: expectedElo,
          onboardingComplete: true,
        },
      });
    }
  });

  it("should return 403 when onboarding already completed", async () => {
    const completedUser = {
      ...mockUser,
      onboardingComplete: true, // Already completed
    };

    req.user = completedUser;

    await completeOnboarding(
      req as AuthenticatedRequest,
      res as Response,
      next,
    );

    expect(res.status).toHaveBeenCalledWith(403);
    expect(res.json).toHaveBeenCalledWith({
      error: "Onboarding has already been completed.",
    });
    expect(mockDatabase.user.update).not.toHaveBeenCalled();
  });

  it("should return 409 when username is already taken by another user", async () => {
    const userWithTakenUsername = {
      id: "different_user_id",
      email: "other@example.com",
      username: validUsername,
      verified: true,
      onboardingComplete: true,
      elo: 500,
      createdAt: new Date(),
      updatedAt: new Date(),
    };

    mockDatabase.user.findUnique.mockResolvedValueOnce(userWithTakenUsername);

    await completeOnboarding(
      req as AuthenticatedRequest,
      res as Response,
      next,
    );

    expect(res.status).toHaveBeenCalledWith(409);
    expect(res.json).toHaveBeenCalledWith({
      error: "This username is already taken.",
    });
    expect(mockDatabase.user.update).not.toHaveBeenCalled();
  });

  it("should allow user to keep their existing username", async () => {
    const existingUser = {
      ...mockUser,
      username: validUsername,
    };

    req.user = existingUser;

    mockDatabase.user.findUnique.mockResolvedValueOnce(existingUser);

    const updatedUser = {
      ...existingUser,
      elo: 500,
      onboardingComplete: true,
    };

    mockDatabase.user.update.mockResolvedValue(updatedUser);

    await completeOnboarding(
      req as AuthenticatedRequest,
      res as Response,
      next,
    );

    expect(res.json).toHaveBeenCalledWith(updatedUser);
    expect(mockDatabase.user.update).toHaveBeenCalledWith({
      where: { id: userId },
      data: {
        username: validUsername,
        elo: 500,
        onboardingComplete: true,
      },
    });
  });

  it("should handle invalid skill level", async () => {
    req.body = { username: validUsername, skillLevel: "expert" }; // Invalid skill level

    await completeOnboarding(
      req as AuthenticatedRequest,
      res as Response,
      next,
    );

    expect(next).toHaveBeenCalledWith(
      expect.objectContaining({
        name: "ZodError",
      }),
    );
  });

  it("should handle invalid username formats", async () => {
    const invalidUsernames = [
      { username: "ab", description: "too short" },
      { username: "a".repeat(33), description: "too long" },
      { username: "user@name", description: "contains invalid character" },
      { username: "user-name", description: "contains dash" },
      { username: "user name", description: "contains space" },
      { username: "", description: "empty string" },
    ];

    for (const { username } of invalidUsernames) {
      vi.clearAllMocks();
      req.body = { username, skillLevel: validSkillLevel };

      await completeOnboarding(
        req as AuthenticatedRequest,
        res as Response,
        next,
      );

      expect(next).toHaveBeenCalledWith(
        expect.objectContaining({
          name: "ZodError",
        }),
      );
    }
  });

  it("should handle valid username formats", async () => {
    const validUsernames = [
      "abc", // minimum length
      "user123",
      "User_Name_123",
      "USERNAME",
      "user_",
      "_user",
      "1234567890",
      "a".repeat(32), // maximum length
    ];

    for (const username of validUsernames) {
      vi.clearAllMocks();

      req.body = { username, skillLevel: validSkillLevel };

      mockDatabase.user.findUnique.mockResolvedValueOnce(null);
      mockDatabase.user.update.mockResolvedValue({
        ...mockUser,
        username,
        elo: 500,
        onboardingComplete: true,
      });

      await completeOnboarding(
        req as AuthenticatedRequest,
        res as Response,
        next,
      );

      expect(mockDatabase.user.update).toHaveBeenCalledWith({
        where: { id: userId },
        data: {
          username,
          elo: 500,
          onboardingComplete: true,
        },
      });

      expect(next).not.toHaveBeenCalled();
    }
  });

  it("should handle missing request body fields", async () => {
    const testCases = [
      { body: { username: validUsername }, description: "missing skill level" },
      {
        body: { skillLevel: validSkillLevel },
        description: "missing username",
      },
      { body: {}, description: "missing both" },
    ];

    for (const { body } of testCases) {
      vi.clearAllMocks();
      req.body = body;

      await completeOnboarding(
        req as AuthenticatedRequest,
        res as Response,
        next,
      );

      expect(next).toHaveBeenCalledWith(
        expect.objectContaining({
          name: "ZodError",
        }),
      );
    }
  });

  it("should handle database errors gracefully", async () => {
    const error = new Error("Database connection failed");
    mockDatabase.user.findUnique.mockRejectedValue(error);

    await completeOnboarding(
      req as AuthenticatedRequest,
      res as Response,
      next,
    );

    expect(next).toHaveBeenCalledWith(error);
    expect(res.status).not.toHaveBeenCalled();
    expect(res.json).not.toHaveBeenCalled();
  });

  it("should handle username check database error", async () => {
    const error = new Error("Username check failed");

    mockDatabase.user.findUnique.mockRejectedValueOnce(error);

    await completeOnboarding(
      req as AuthenticatedRequest,
      res as Response,
      next,
    );

    expect(next).toHaveBeenCalledWith(error);
  });

  it("should handle user update database error", async () => {
    const error = new Error("User update failed");

    mockDatabase.user.findUnique.mockResolvedValueOnce(null);
    mockDatabase.user.update.mockRejectedValue(error);

    await completeOnboarding(
      req as AuthenticatedRequest,
      res as Response,
      next,
    );

    expect(next).toHaveBeenCalledWith(error);
  });
});
