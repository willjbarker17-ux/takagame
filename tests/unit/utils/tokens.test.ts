import { describe, it, expect, vi, beforeEach } from "vitest";
import jwt from "jsonwebtoken";
import ms from "ms";
import {
  createAccessToken,
  createRefreshToken,
  verifyToken,
  type Payload,
} from "@/utils/tokens";
import config from "@/config";

describe("tokens", () => {
  const mockUserId = "test-user-id";

  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe("createAccessToken", () => {
    it("should create a valid access token with correct payload", () => {
      const token = createAccessToken(mockUserId);

      expect(token).toBeDefined();
      expect(typeof token).toBe("string");

      const decoded = jwt.verify(token, config.JWT_SECRET) as Payload;
      expect(decoded.userId).toBe(mockUserId);
      expect(decoded.iat).toBeDefined();
      expect(decoded.exp).toBeDefined();
    });

    it("should create token with correct expiration time", () => {
      const token = createAccessToken(mockUserId);
      const decoded = jwt.verify(token, config.JWT_SECRET) as Payload;

      const expectedExp =
        Math.floor(Date.now() / 1000) +
        Math.floor(ms(config.JWT_EXPIRES_IN) / 1000);
      expect(decoded.exp).toBeCloseTo(expectedExp, -1);
    });
  });

  describe("createRefreshToken", () => {
    it("should create a valid refresh token with correct payload", () => {
      const token = createRefreshToken(mockUserId);

      expect(token).toBeDefined();
      expect(typeof token).toBe("string");

      const decoded = jwt.verify(token, config.JWT_SECRET) as Payload;
      expect(decoded.userId).toBe(mockUserId);
      expect(decoded.iat).toBeDefined();
      expect(decoded.exp).toBeDefined();
    });

    it("should create token with correct expiration time", () => {
      const token = createRefreshToken(mockUserId);
      const decoded = jwt.verify(token, config.JWT_SECRET) as Payload;

      const expectedExp =
        Math.floor(Date.now() / 1000) +
        Math.floor(ms(config.JWT_REFRESH_EXPIRES_IN) / 1000);
      expect(decoded.exp).toBeCloseTo(expectedExp, -1);
    });
  });

  describe("verifyToken", () => {
    it("should verify a valid token and return payload", () => {
      const token = createAccessToken(mockUserId);
      const payload = verifyToken(token);

      expect(payload).not.toBeNull();
      expect(payload?.userId).toBe(mockUserId);
      expect(payload?.iat).toBeDefined();
      expect(payload?.exp).toBeDefined();
    });

    it("should return null for invalid token", () => {
      const invalidToken = "invalid.token.here";
      const payload = verifyToken(invalidToken);

      expect(payload).toBeNull();
    });

    it("should return null for expired token", () => {
      const expiredToken = jwt.sign({ userId: mockUserId }, config.JWT_SECRET, {
        expiresIn: "-1s",
      });

      const payload = verifyToken(expiredToken);
      expect(payload).toBeNull();
    });

    it("should return null for token with wrong signature", () => {
      const tokenWithWrongSignature = jwt.sign(
        { userId: mockUserId },
        "wrong-secret",
        { expiresIn: "1h" },
      );

      const payload = verifyToken(tokenWithWrongSignature);
      expect(payload).toBeNull();
    });
  });

  describe("token integration", () => {
    it("should create and verify access token successfully", () => {
      const accessToken = createAccessToken(mockUserId);
      const payload = verifyToken(accessToken);

      expect(payload).not.toBeNull();
      expect(payload?.userId).toBe(mockUserId);
    });

    it("should create and verify refresh token successfully", () => {
      const refreshToken = createRefreshToken(mockUserId);
      const payload = verifyToken(refreshToken);

      expect(payload).not.toBeNull();
      expect(payload?.userId).toBe(mockUserId);
    });
  });
});
