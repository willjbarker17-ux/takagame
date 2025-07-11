import { describe, it, expect, vi, beforeEach } from "vitest";
import mockDatabase from "@tests/mocks/database";

vi.mock("@/database", () => ({ default: mockDatabase }));

import type { Request, Response, NextFunction } from "express";
import {
  emailRequestOtp,
  emailVerifyOtp,
  logout,
  refreshToken,
} from "@/controllers/auth.controller";
import prisma from "@tests/mocks/database";
import { sendMagicLinkEmail } from "@/utils/email";
import {
  createAccessToken,
  createRefreshToken,
  verifyToken,
} from "@/utils/tokens";
import config from "@/config";

vi.mock("@/utils/email", () => ({
  sendMagicLinkEmail: vi.fn(),
}));

vi.mock("@/utils/tokens", () => ({
  createAccessToken: vi.fn(),
  createRefreshToken: vi.fn(),
  verifyToken: vi.fn(),
}));

const mockCreateAccessToken = vi.mocked(createAccessToken);
const mockCreateRefreshToken = vi.mocked(createRefreshToken);
const mockVerifyToken = vi.mocked(verifyToken);

type EmailOtpRequestBody = {
  email?: string;
};

describe("auth.controller: emailRequestOtp", () => {
  let req: Partial<Request & { body: EmailOtpRequestBody }>;
  let res: Partial<Response>;
  let next: NextFunction;

  beforeEach(() => {
    req = {
      body: {},
    };
    res = {
      sendStatus: vi.fn(),
    };
    next = vi.fn();
    vi.clearAllMocks();
  });

  it("should create a verification token and send an email for a valid request", async () => {
    const userEmail = "test@example.com";
    const verificationCode = "test_verification_code";
    (req.body as EmailOtpRequestBody).email = userEmail;

    prisma.verificationToken.create.mockResolvedValue({
      code: verificationCode,
      email: userEmail,
      expiresAt: new Date(Date.now() + 15 * 60 * 1000),
      createdAt: new Date(),
    });

    vi.mocked(sendMagicLinkEmail).mockResolvedValue(undefined);

    await emailRequestOtp(req as Request, res as Response, next);

    expect(prisma.verificationToken.create).toHaveBeenCalledWith({
      data: {
        email: userEmail,
        expiresAt: expect.any(Date) as Date,
      },
    });
    expect(sendMagicLinkEmail).toHaveBeenCalledWith(
      userEmail,
      verificationCode,
    );
    expect(res.sendStatus).toHaveBeenCalledWith(200);
    expect(next).not.toHaveBeenCalled();
  });

  it("should call next with a Zod error for an invalid email", async () => {
    (req.body as EmailOtpRequestBody).email = "not-an-email";

    await emailRequestOtp(req as Request, res as Response, next);

    expect(next).toHaveBeenCalledWith(expect.any(Error));
    expect(next).toHaveBeenCalledWith(
      expect.objectContaining({ name: "ZodError" }),
    );
    expect(prisma.verificationToken.create).not.toHaveBeenCalled();
    expect(sendMagicLinkEmail).not.toHaveBeenCalled();
  });

  it("should call next with a Zod error for a missing email", async () => {
    delete (req.body as EmailOtpRequestBody).email;

    await emailRequestOtp(req as Request, res as Response, next);

    expect(next).toHaveBeenCalledWith(expect.any(Error));
    expect(next).toHaveBeenCalledWith(
      expect.objectContaining({ name: "ZodError" }),
    );
  });

  it("should call next with the error if database creation fails", async () => {
    const dbError = new Error("Database connection failed");
    (req.body as EmailOtpRequestBody).email = "test@example.com";
    prisma.verificationToken.create.mockRejectedValue(dbError);

    await emailRequestOtp(req as Request, res as Response, next);

    expect(next).toHaveBeenCalledWith(dbError);
    expect(res.sendStatus).not.toHaveBeenCalled();
  });

  it("should call next with the error if email sending fails", async () => {
    const emailError = new Error("SMTP server down");
    (req.body as EmailOtpRequestBody).email = "test@example.com";
    prisma.verificationToken.create.mockResolvedValue({
      code: "test_code",
      email: "test@example.com",
      expiresAt: new Date(),
      createdAt: new Date(),
    });
    vi.mocked(sendMagicLinkEmail).mockRejectedValue(emailError);

    await emailRequestOtp(req as Request, res as Response, next);

    expect(next).toHaveBeenCalledWith(emailError);
    expect(res.sendStatus).not.toHaveBeenCalled();
  });
});

describe("auth.controller: emailVerifyOtp", () => {
  let req: Partial<Request>;
  let res: Partial<Response>;
  let next: NextFunction;

  const validCode = "cuid_valid_code";
  const validEmail = "test@example.com";
  const userId = "user_123";

  beforeEach(() => {
    req = {
      body: { code: validCode },
      cookies: {},
      headers: {},
    };
    res = {
      status: vi.fn().mockReturnThis(),
      json: vi.fn().mockReturnThis(),
      sendStatus: vi.fn().mockReturnThis(),
      cookie: vi.fn().mockReturnThis(),
      clearCookie: vi.fn().mockReturnThis(),
    };
    next = vi.fn();

    mockCreateAccessToken.mockReturnValue("mock_access_token");
    mockCreateRefreshToken.mockReturnValue("mock_refresh_token");

    vi.clearAllMocks();
  });

  it("should verify OTP and create new user when user doesn't exist", async () => {
    const verificationToken = {
      code: validCode,
      email: validEmail,
      expiresAt: new Date(Date.now() + 10 * 60 * 1000),
      createdAt: new Date(),
    };

    const newUser = {
      id: userId,
      email: validEmail,
      username: `user_${Date.now()}`,
      verified: true,
      onboardingComplete: false,
      elo: 200,
      createdAt: new Date(),
      updatedAt: new Date(),
    };

    prisma.verificationToken.findFirst.mockResolvedValue(verificationToken);
    prisma.user.findUnique.mockResolvedValue(null);
    prisma.user.create.mockResolvedValue(newUser);
    prisma.verificationToken.delete.mockResolvedValue(verificationToken);
    prisma.refreshToken.create.mockResolvedValue({
      id: "refresh_123",
      token: "mock_refresh_token",
      userId,
      userAgent: "",
      createdAt: new Date(),
    });

    await emailVerifyOtp(req as Request, res as Response, next);

    expect(prisma.user.create).toHaveBeenCalledWith({
      data: {
        email: validEmail,
        username: expect.stringMatching(/^user_\d+$/) as string,
        verified: true,
        onboardingComplete: false,
      },
    });

    expect(prisma.verificationToken.delete).toHaveBeenCalledWith({
      where: { code: validCode },
    });

    expect(mockCreateAccessToken).toHaveBeenCalledWith(userId);
    expect(mockCreateRefreshToken).toHaveBeenCalledWith(userId);

    expect(res.cookie).toHaveBeenCalledWith(
      config.JWT_REFRESH_TOKEN_COOKIE_NAME,
      "mock_refresh_token",
      expect.objectContaining({
        httpOnly: true,
        sameSite: "lax",
      }),
    );

    expect(res.json).toHaveBeenCalledWith({
      accessToken: "mock_access_token",
      needsOnboarding: true,
    });
  });

  it("should verify OTP and login existing verified user", async () => {
    const verificationToken = {
      code: validCode,
      email: validEmail,
      expiresAt: new Date(Date.now() + 10 * 60 * 1000),
      createdAt: new Date(),
    };

    const existingUser = {
      id: userId,
      email: validEmail,
      username: "existing_user",
      verified: true,
      onboardingComplete: true,
      elo: 500,
      createdAt: new Date(),
      updatedAt: new Date(),
    };

    prisma.verificationToken.findFirst.mockResolvedValue(verificationToken);
    prisma.user.findUnique.mockResolvedValue(existingUser);
    prisma.verificationToken.delete.mockResolvedValue(verificationToken);
    prisma.refreshToken.create.mockResolvedValue({
      id: "refresh_123",
      token: "mock_refresh_token",
      userId,
      userAgent: "",
      createdAt: new Date(),
    });

    await emailVerifyOtp(req as Request, res as Response, next);

    expect(prisma.user.create).not.toHaveBeenCalled();
    expect(prisma.user.update).not.toHaveBeenCalled();

    expect(res.json).toHaveBeenCalledWith({
      accessToken: "mock_access_token",
      needsOnboarding: false,
    });
  });

  it("should verify unverified existing user", async () => {
    const verificationToken = {
      code: validCode,
      email: validEmail,
      expiresAt: new Date(Date.now() + 10 * 60 * 1000),
      createdAt: new Date(),
    };

    const unverifiedUser = {
      id: userId,
      email: validEmail,
      username: "existing_user",
      verified: false,
      onboardingComplete: false,
      elo: 200,
      createdAt: new Date(),
      updatedAt: new Date(),
    };

    const verifiedUser = {
      ...unverifiedUser,
      verified: true,
    };

    prisma.verificationToken.findFirst.mockResolvedValue(verificationToken);
    prisma.user.findUnique.mockResolvedValue(unverifiedUser);
    prisma.user.update.mockResolvedValue(verifiedUser);
    prisma.verificationToken.delete.mockResolvedValue(verificationToken);
    prisma.refreshToken.create.mockResolvedValue({
      id: "refresh_123",
      token: "mock_refresh_token",
      userId,
      userAgent: "",
      createdAt: new Date(),
    });

    await emailVerifyOtp(req as Request, res as Response, next);

    expect(prisma.user.update).toHaveBeenCalledWith({
      where: { id: userId },
      data: { verified: true },
    });

    expect(res.json).toHaveBeenCalledWith({
      accessToken: "mock_access_token",
      needsOnboarding: true,
    });
  });

  it("should handle existing refresh token for same user", async () => {
    const verificationToken = {
      code: validCode,
      email: validEmail,
      expiresAt: new Date(Date.now() + 10 * 60 * 1000),
      createdAt: new Date(),
    };

    const existingUser = {
      id: userId,
      email: validEmail,
      username: "existing_user",
      verified: true,
      onboardingComplete: true,
      elo: 500,
      createdAt: new Date(),
      updatedAt: new Date(),
    };

    const existingRefreshToken = {
      id: "refresh_123",
      token: "existing_refresh_token",
      userId,
      userAgent: "",
      createdAt: new Date(),
    };

    req.cookies = {
      [config.JWT_REFRESH_TOKEN_COOKIE_NAME]: "existing_refresh_token",
    };

    prisma.verificationToken.findFirst.mockResolvedValue(verificationToken);
    prisma.user.findUnique.mockResolvedValue(existingUser);
    prisma.refreshToken.findUnique.mockResolvedValue(existingRefreshToken);
    prisma.refreshToken.delete.mockResolvedValue(existingRefreshToken);
    prisma.verificationToken.delete.mockResolvedValue(verificationToken);
    prisma.refreshToken.create.mockResolvedValue({
      id: "refresh_456",
      token: "mock_refresh_token",
      userId,
      userAgent: "",
      createdAt: new Date(),
    });

    await emailVerifyOtp(req as Request, res as Response, next);

    expect(prisma.refreshToken.delete).toHaveBeenCalledWith({
      where: { token: "existing_refresh_token" },
    });

    expect(res.clearCookie).toHaveBeenCalledWith(
      config.JWT_REFRESH_TOKEN_COOKIE_NAME,
      expect.objectContaining({
        httpOnly: true,
        sameSite: "lax",
      }),
    );
  });

  it("should return 400 when verification token not found", async () => {
    prisma.verificationToken.findFirst.mockResolvedValue(null);

    await emailVerifyOtp(req as Request, res as Response, next);

    expect(res.status).toHaveBeenCalledWith(400);
    expect(res.json).toHaveBeenCalledWith({
      error: "Verification request not found",
    });
  });

  it("should return 400 when verification token is expired", async () => {
    const expiredToken = {
      code: validCode,
      email: validEmail,
      expiresAt: new Date(Date.now() - 10 * 60 * 1000),
      createdAt: new Date(),
    };

    prisma.verificationToken.findFirst.mockResolvedValue(expiredToken);

    await emailVerifyOtp(req as Request, res as Response, next);

    expect(res.status).toHaveBeenCalledWith(400);
    expect(res.json).toHaveBeenCalledWith({
      error: "Login request has expired",
    });
  });

  it("should return validation error for invalid code format", async () => {
    req.body = { code: "invalid_code" };

    await emailVerifyOtp(req as Request, res as Response, next);

    expect(next).toHaveBeenCalledWith(
      expect.objectContaining({
        name: "ZodError",
      }),
    );
  });

  it("should include user agent in refresh token", async () => {
    const verificationToken = {
      code: validCode,
      email: validEmail,
      expiresAt: new Date(Date.now() + 10 * 60 * 1000),
      createdAt: new Date(),
    };

    const newUser = {
      id: userId,
      email: validEmail,
      username: `user_${Date.now()}`,
      verified: true,
      onboardingComplete: false,
      elo: 200,
      createdAt: new Date(),
      updatedAt: new Date(),
    };

    req.headers = { "user-agent": "Mozilla/5.0 Test Browser" };

    prisma.verificationToken.findFirst.mockResolvedValue(verificationToken);
    prisma.user.findUnique.mockResolvedValue(null);
    prisma.user.create.mockResolvedValue(newUser);
    prisma.verificationToken.delete.mockResolvedValue(verificationToken);
    prisma.refreshToken.create.mockResolvedValue({
      id: "refresh_123",
      token: "mock_refresh_token",
      userId,
      userAgent: "Mozilla/5.0 Test Browser",
      createdAt: new Date(),
    });

    await emailVerifyOtp(req as Request, res as Response, next);

    expect(prisma.refreshToken.create).toHaveBeenCalledWith({
      data: {
        token: "mock_refresh_token",
        userId,
        userAgent: "Mozilla/5.0 Test Browser",
      },
    });
  });

  it("should call next with error when database operation fails", async () => {
    const error = new Error("Database connection failed");
    prisma.verificationToken.findFirst.mockRejectedValue(error);

    await emailVerifyOtp(req as Request, res as Response, next);

    expect(next).toHaveBeenCalledWith(error);
  });
});

describe("auth.controller: logout", () => {
  let req: Partial<Request>;
  let res: Partial<Response>;
  let next: NextFunction;

  const validRefreshToken = "valid_refresh_token";

  beforeEach(() => {
    req = {
      body: { refreshToken: validRefreshToken },
    };
    res = {
      sendStatus: vi.fn().mockReturnThis(),
    };
    next = vi.fn();

    vi.clearAllMocks();
  });

  it("should successfully logout user with valid refresh token", async () => {
    const refreshTokenRecord = {
      id: "refresh_123",
      token: validRefreshToken,
      userId: "user_123",
      userAgent: "Mozilla/5.0",
      createdAt: new Date(),
    };

    prisma.refreshToken.delete.mockResolvedValue(refreshTokenRecord);

    await logout(req as Request, res as Response, next);

    expect(prisma.refreshToken.delete).toHaveBeenCalledWith({
      where: { token: validRefreshToken },
    });

    expect(res.sendStatus).toHaveBeenCalledWith(200);
    expect(next).not.toHaveBeenCalled();
  });

  it("should handle missing refreshToken in request body", async () => {
    req.body = {};

    await logout(req as Request, res as Response, next);

    expect(next).toHaveBeenCalledWith(
      expect.objectContaining({
        name: "ZodError",
      }),
    );
  });

  it("should handle empty refreshToken", async () => {
    req.body = { refreshToken: "" };

    await logout(req as Request, res as Response, next);

    expect(next).toHaveBeenCalledWith(
      expect.objectContaining({
        name: "ZodError",
      }),
    );
  });

  it("should handle extra fields in request body", async () => {
    req.body = {
      refreshToken: validRefreshToken,
      extraField: "should not be allowed",
    };

    await logout(req as Request, res as Response, next);

    expect(next).toHaveBeenCalledWith(
      expect.objectContaining({
        name: "ZodError",
      }),
    );
  });

  it("should handle database deletion failure gracefully", async () => {
    const error = new Error("Token not found");
    prisma.refreshToken.delete.mockRejectedValue(error);

    await logout(req as Request, res as Response, next);

    expect(next).toHaveBeenCalledWith(error);
    expect(res.sendStatus).not.toHaveBeenCalled();
  });
});

describe("auth.controller: refreshToken", () => {
  let req: Partial<Request>;
  let res: Partial<Response>;
  let next: NextFunction;

  const validRefreshToken = "valid_refresh_token";
  const userId = "user_123";

  beforeEach(() => {
    req = {
      body: { refreshToken: validRefreshToken },
    };
    res = {
      status: vi.fn().mockReturnThis(),
      json: vi.fn().mockReturnThis(),
    };
    next = vi.fn();

    mockCreateAccessToken.mockReturnValue("new_access_token");

    vi.clearAllMocks();
  });

  it("should successfully refresh token with valid refresh token", async () => {
    const refreshTokenRecord = {
      id: "refresh_123",
      token: validRefreshToken,
      userId,
      userAgent: "Mozilla/5.0",
      createdAt: new Date(),
    };

    const tokenPayload = {
      userId,
      iat: Math.floor(Date.now() / 1000),
      exp: Math.floor(Date.now() / 1000) + 3600,
    };

    prisma.refreshToken.findUnique.mockResolvedValue(refreshTokenRecord);
    mockVerifyToken.mockReturnValue(tokenPayload);

    await refreshToken(req as Request, res as Response, next);

    expect(prisma.refreshToken.findUnique).toHaveBeenCalledWith({
      where: { token: validRefreshToken },
    });

    expect(mockVerifyToken).toHaveBeenCalledWith(validRefreshToken);
    expect(mockCreateAccessToken).toHaveBeenCalledWith(userId);

    expect(res.json).toHaveBeenCalledWith({
      accessToken: "new_access_token",
    });

    expect(next).not.toHaveBeenCalled();
  });

  it("should return 401 when refresh token not found in database", async () => {
    const tokenPayload = {
      userId,
      iat: Math.floor(Date.now() / 1000),
      exp: Math.floor(Date.now() / 1000) + 3600,
    };

    prisma.refreshToken.findUnique.mockResolvedValue(null);
    mockVerifyToken.mockReturnValue(tokenPayload);
    prisma.refreshToken.deleteMany.mockResolvedValue({ count: 2 });

    await refreshToken(req as Request, res as Response, next);

    expect(prisma.refreshToken.findUnique).toHaveBeenCalledWith({
      where: { token: validRefreshToken },
    });

    expect(mockVerifyToken).toHaveBeenCalledWith(validRefreshToken);

    // Should revoke all tokens for security
    expect(prisma.refreshToken.deleteMany).toHaveBeenCalledWith({
      where: { userId },
    });

    expect(res.status).toHaveBeenCalledWith(401);
    expect(res.json).toHaveBeenCalledWith({
      error: "Invalid refresh token",
    });
  });

  it("should return 401 when refresh token not found and JWT is invalid", async () => {
    prisma.refreshToken.findUnique.mockResolvedValue(null);
    mockVerifyToken.mockReturnValue(null);

    await refreshToken(req as Request, res as Response, next);

    expect(prisma.refreshToken.findUnique).toHaveBeenCalledWith({
      where: { token: validRefreshToken },
    });

    expect(mockVerifyToken).toHaveBeenCalledWith(validRefreshToken);

    // Should not revoke tokens if JWT is invalid
    expect(prisma.refreshToken.deleteMany).not.toHaveBeenCalled();

    expect(res.status).toHaveBeenCalledWith(401);
    expect(res.json).toHaveBeenCalledWith({
      error: "Invalid refresh token",
    });
  });

  it("should return 401 when JWT payload is invalid", async () => {
    const refreshTokenRecord = {
      id: "refresh_123",
      token: validRefreshToken,
      userId,
      userAgent: "Mozilla/5.0",
      createdAt: new Date(),
    };

    prisma.refreshToken.findUnique.mockResolvedValue(refreshTokenRecord);
    mockVerifyToken.mockReturnValue(null);

    await refreshToken(req as Request, res as Response, next);

    expect(res.status).toHaveBeenCalledWith(401);
    expect(res.json).toHaveBeenCalledWith({
      error: "Invalid refresh token",
    });
  });

  it("should return 401 when JWT userId doesn't match database record", async () => {
    const refreshTokenRecord = {
      id: "refresh_123",
      token: validRefreshToken,
      userId,
      userAgent: "Mozilla/5.0",
      createdAt: new Date(),
    };

    const tokenPayload = {
      userId: "different_user_123", // Different user ID
      iat: Math.floor(Date.now() / 1000),
      exp: Math.floor(Date.now() / 1000) + 3600,
    };

    prisma.refreshToken.findUnique.mockResolvedValue(refreshTokenRecord);
    mockVerifyToken.mockReturnValue(tokenPayload);

    await refreshToken(req as Request, res as Response, next);

    expect(res.status).toHaveBeenCalledWith(401);
    expect(res.json).toHaveBeenCalledWith({
      error: "Invalid refresh token",
    });
  });

  it("should handle missing refreshToken in request body", async () => {
    req.body = {};

    await refreshToken(req as Request, res as Response, next);

    expect(next).toHaveBeenCalledWith(
      expect.objectContaining({
        name: "ZodError",
      }),
    );
  });

  it("should handle empty refreshToken", async () => {
    req.body = { refreshToken: "" };

    await refreshToken(req as Request, res as Response, next);

    expect(next).toHaveBeenCalledWith(
      expect.objectContaining({
        name: "ZodError",
      }),
    );
  });

  it("should handle extra fields in request body", async () => {
    req.body = {
      refreshToken: validRefreshToken,
      extraField: "should not be allowed",
    };

    await refreshToken(req as Request, res as Response, next);

    expect(next).toHaveBeenCalledWith(
      expect.objectContaining({
        name: "ZodError",
      }),
    );
  });

  it("should handle database lookup failure gracefully", async () => {
    const error = new Error("Database connection failed");
    prisma.refreshToken.findUnique.mockRejectedValue(error);

    await refreshToken(req as Request, res as Response, next);

    expect(next).toHaveBeenCalledWith(error);
    expect(res.status).not.toHaveBeenCalled();
    expect(res.json).not.toHaveBeenCalled();
  });

  it("should handle token revocation failure when token is stolen", async () => {
    const tokenPayload = {
      userId,
      iat: Math.floor(Date.now() / 1000),
      exp: Math.floor(Date.now() / 1000) + 3600,
    };

    const deleteError = new Error("Database deletion failed");

    prisma.refreshToken.findUnique.mockResolvedValue(null);
    mockVerifyToken.mockReturnValue(tokenPayload);
    prisma.refreshToken.deleteMany.mockRejectedValue(deleteError);

    await refreshToken(req as Request, res as Response, next);

    expect(next).toHaveBeenCalledWith(deleteError);
  });

  it("should handle JWT verification with minimal payload", async () => {
    const refreshTokenRecord = {
      id: "refresh_123",
      token: validRefreshToken,
      userId,
      userAgent: "Mozilla/5.0",
      createdAt: new Date(),
    };

    const minimalPayload = {
      userId,
      // Missing iat and exp (optional fields)
    };

    prisma.refreshToken.findUnique.mockResolvedValue(refreshTokenRecord);
    mockVerifyToken.mockReturnValue(minimalPayload);

    await refreshToken(req as Request, res as Response, next);

    expect(mockCreateAccessToken).toHaveBeenCalledWith(userId);
    expect(res.json).toHaveBeenCalledWith({
      accessToken: "new_access_token",
    });
  });
});
