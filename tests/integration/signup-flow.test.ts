import { describe, it, expect, beforeEach, vi } from "vitest";
import supertest from "supertest";
import { createTestApp } from "@tests/helpers/testApp";
import mockDatabase from "@tests/mocks/database";
import { sendMagicLinkEmail } from "@/utils/email";

vi.mock("@/utils/email", () => ({
  sendMagicLinkEmail: vi.fn(),
}));

const mockSendMagicLinkEmail = vi.mocked(sendMagicLinkEmail);

describe("Signup Flow Integration Tests", () => {
  const app = createTestApp();
  const request = supertest(app);

  const testEmail = "test@example.com";
  const testUsername = "testuser123";
  const testSkillLevel = "intermediate";

  beforeEach(() => {
    vi.clearAllMocks();
    mockSendMagicLinkEmail.mockResolvedValue(undefined);
  });

  describe("Happy Path: Complete Signup Flow", () => {
    it("should handle complete signup flow: request -> verify -> onboard -> logout -> login", async () => {
      const verificationCode = "clg0p72z4000008ld12345678";
      const userId = "user_123";

      // Step 1: Request magic link
      const verificationToken = {
        code: verificationCode,
        email: testEmail,
        expiresAt: new Date(Date.now() + 15 * 60 * 1000),
        createdAt: new Date(),
      };

      mockDatabase.verificationToken.create.mockResolvedValue(
        verificationToken,
      );

      await request
        .post("/auth/magic-link")
        .send({ email: testEmail })
        .expect(200);

      expect(mockDatabase.verificationToken.create).toHaveBeenCalledWith({
        data: {
          email: testEmail,
          expiresAt: expect.any(Date) as Date,
        },
      });
      expect(mockSendMagicLinkEmail).toHaveBeenCalledWith(
        testEmail,
        verificationCode,
      );

      // Step 2: Verify magic link (new user signup)
      const newUser = {
        id: userId,
        email: testEmail,
        username: `user_${Date.now()}`,
        verified: true,
        onboardingComplete: false,
        elo: 200,
        createdAt: new Date(),
        updatedAt: new Date(),
      };

      mockDatabase.verificationToken.findFirst.mockResolvedValue(
        verificationToken,
      );
      mockDatabase.user.findUnique.mockResolvedValue(null); // New user
      mockDatabase.user.create.mockResolvedValue(newUser);
      mockDatabase.verificationToken.delete.mockResolvedValue(
        verificationToken,
      );
      mockDatabase.refreshToken.create.mockResolvedValue({
        id: "refresh_123",
        token: "mock_refresh_token",
        userId,
        userAgent: "",
        createdAt: new Date(),
      });

      const verifyResponse = await request
        .post("/auth/verify")
        .send({ code: verificationCode })
        .expect(200);

      expect(verifyResponse.body).toEqual({
        accessToken: expect.any(String) as string,
        needsOnboarding: true,
      });

      const { accessToken } = verifyResponse.body as { accessToken: string };
      const cookies = verifyResponse.headers["set-cookie"];
      expect(cookies).toBeDefined();

      // Step 3: Complete onboarding
      const updatedUser = {
        ...newUser,
        username: testUsername,
        elo: 500, // intermediate
        onboardingComplete: true,
      };

      const expectedUserResponse = {
        ...updatedUser,
        createdAt: expect.any(String) as string,
        updatedAt: expect.any(String) as string,
      };

      // requireUser middleware will fetch the user from database
      mockDatabase.user.findUnique.mockResolvedValueOnce(newUser); // requireUser middleware
      mockDatabase.user.findUnique.mockResolvedValueOnce(null); // Username availability check
      mockDatabase.user.update.mockResolvedValue(updatedUser);

      const onboardingResponse = await request
        .post("/onboarding/complete")
        .set("Authorization", `Bearer ${accessToken}`)
        .send({
          username: testUsername,
          skillLevel: testSkillLevel,
        })
        .expect(200);

      expect(onboardingResponse.body).toEqual(expectedUserResponse);

      // Step 4: Logout
      mockDatabase.refreshToken.delete.mockResolvedValue({
        id: "refresh_123",
        token: "mock_refresh_token",
        userId,
        userAgent: "",
        createdAt: new Date(),
      });

      await request
        .get("/auth/logout")
        .send({ refreshToken: "mock_refresh_token" })
        .expect(200);

      // Step 5: Login again (existing user)
      const loginVerificationCode = "clg0p72z4000008ld98765432";
      const loginVerificationToken = {
        code: loginVerificationCode,
        email: testEmail,
        expiresAt: new Date(Date.now() + 15 * 60 * 1000),
        createdAt: new Date(),
      };

      mockDatabase.verificationToken.create.mockResolvedValue(
        loginVerificationToken,
      );

      await request
        .post("/auth/magic-link")
        .send({ email: testEmail })
        .expect(200);

      // Verify login
      mockDatabase.verificationToken.findFirst.mockResolvedValue(
        loginVerificationToken,
      );
      mockDatabase.user.findUnique.mockResolvedValue(updatedUser); // Existing completed user
      mockDatabase.verificationToken.delete.mockResolvedValue(
        loginVerificationToken,
      );
      mockDatabase.refreshToken.create.mockResolvedValue({
        id: "refresh_456",
        token: "new_refresh_token",
        userId,
        userAgent: "",
        createdAt: new Date(),
      });

      const loginResponse = await request
        .post("/auth/verify")
        .send({ code: loginVerificationCode })
        .expect(200);

      expect(loginResponse.body).toEqual({
        accessToken: expect.any(String) as string,
        needsOnboarding: false, // Already completed
      });
    });
  });

  describe("Edge Case: Session Expiration Between Verification and Onboarding", () => {
    it("should handle verification -> session expire -> login -> onboarding", async () => {
      const verificationCode = "clg0p72z4000008ld12345678";
      const userId = "user_123";

      // Step 1: Request magic link and verify (create new user)
      const verificationToken = {
        code: verificationCode,
        email: testEmail,
        expiresAt: new Date(Date.now() + 15 * 60 * 1000),
        createdAt: new Date(),
      };

      const newUser = {
        id: userId,
        email: testEmail,
        username: `user_${Date.now()}`,
        verified: true,
        onboardingComplete: false,
        elo: 200,
        createdAt: new Date(),
        updatedAt: new Date(),
      };

      mockDatabase.verificationToken.create.mockResolvedValue(
        verificationToken,
      );
      mockDatabase.verificationToken.findFirst.mockResolvedValue(
        verificationToken,
      );
      mockDatabase.user.findUnique.mockResolvedValue(null);
      mockDatabase.user.create.mockResolvedValue(newUser);
      mockDatabase.verificationToken.delete.mockResolvedValue(
        verificationToken,
      );
      mockDatabase.refreshToken.create.mockResolvedValue({
        id: "refresh_123",
        token: "mock_refresh_token",
        userId,
        userAgent: "",
        createdAt: new Date(),
      });

      await request
        .post("/auth/magic-link")
        .send({ email: testEmail })
        .expect(200);

      const verifyResponse = await request
        .post("/auth/verify")
        .send({ code: verificationCode })
        .expect(200);

      expect(
        (verifyResponse.body as { needsOnboarding: boolean }).needsOnboarding,
      ).toBe(true);

      // Step 2: Simulate session expiration by trying onboarding with expired/invalid token
      await request
        .post("/onboarding/complete")
        .set("Authorization", "Bearer expired_or_invalid_token")
        .send({
          username: testUsername,
          skillLevel: testSkillLevel,
        })
        .expect(403); // Should be unauthorized

      // Step 3: User needs to login again
      const loginVerificationCode = "clg0p72z4000008ld98765432";
      const loginVerificationToken = {
        code: loginVerificationCode,
        email: testEmail,
        expiresAt: new Date(Date.now() + 15 * 60 * 1000),
        createdAt: new Date(),
      };

      mockDatabase.verificationToken.create.mockResolvedValue(
        loginVerificationToken,
      );
      mockDatabase.verificationToken.findFirst.mockResolvedValue(
        loginVerificationToken,
      );
      mockDatabase.user.findUnique.mockResolvedValue(newUser); // User exists but onboarding incomplete
      mockDatabase.verificationToken.delete.mockResolvedValue(
        loginVerificationToken,
      );
      mockDatabase.refreshToken.create.mockResolvedValue({
        id: "refresh_456",
        token: "new_refresh_token",
        userId,
        userAgent: "",
        createdAt: new Date(),
      });

      await request
        .post("/auth/magic-link")
        .send({ email: testEmail })
        .expect(200);

      const loginResponse = await request
        .post("/auth/verify")
        .send({ code: loginVerificationCode })
        .expect(200);

      expect(
        (loginResponse.body as { needsOnboarding: boolean }).needsOnboarding,
      ).toBe(true); // Still needs onboarding

      // Step 4: Complete onboarding with new valid token
      const updatedUser = {
        ...newUser,
        username: testUsername,
        elo: 500,
        onboardingComplete: true,
      };

      const expectedUserResponse = {
        ...updatedUser,
        createdAt: expect.any(String) as string,
        updatedAt: expect.any(String) as string,
      };

      // requireUser middleware will fetch the user from database
      mockDatabase.user.findUnique.mockResolvedValueOnce(newUser); // requireUser middleware
      mockDatabase.user.findUnique.mockResolvedValueOnce(null); // Username availability check
      mockDatabase.user.update.mockResolvedValue(updatedUser);

      const onboardingResponse = await request
        .post("/onboarding/complete")
        .set(
          "Authorization",
          `Bearer ${(loginResponse.body as { accessToken: string }).accessToken}`,
        )
        .send({
          username: testUsername,
          skillLevel: testSkillLevel,
        })
        .expect(200);
      expect(onboardingResponse.body).toEqual(expectedUserResponse);
    });
  });

  describe("Edge Case: Multiple Verification Attempts", () => {
    it("should handle expired verification token and require new magic link", async () => {
      const expiredCode = "clg0p72z4000008ld11112222";

      // Step 1: Try to verify with expired token
      const expiredToken = {
        code: expiredCode,
        email: testEmail,
        expiresAt: new Date(Date.now() - 10 * 60 * 1000), // 10 minutes ago
        createdAt: new Date(),
      };

      mockDatabase.verificationToken.findFirst.mockResolvedValue(expiredToken);

      await request
        .post("/auth/verify")
        .send({ code: expiredCode })
        .expect(400);

      // Step 2: Request new magic link
      const newVerificationCode = "clg0p72z4000008ld33334444";
      const newVerificationToken = {
        code: newVerificationCode,
        email: testEmail,
        expiresAt: new Date(Date.now() + 15 * 60 * 1000),
        createdAt: new Date(),
      };

      mockDatabase.verificationToken.create.mockResolvedValue(
        newVerificationToken,
      );

      await request
        .post("/auth/magic-link")
        .send({ email: testEmail })
        .expect(200);

      // Step 3: Verify with new valid token
      const userId = "user_123";
      const newUser = {
        id: userId,
        email: testEmail,
        username: `user_${Date.now()}`,
        verified: true,
        onboardingComplete: false,
        elo: 200,
        createdAt: new Date(),
        updatedAt: new Date(),
      };

      mockDatabase.verificationToken.findFirst.mockResolvedValue(
        newVerificationToken,
      );
      mockDatabase.user.findUnique.mockResolvedValue(null);
      mockDatabase.user.create.mockResolvedValue(newUser);
      mockDatabase.verificationToken.delete.mockResolvedValue(
        newVerificationToken,
      );
      mockDatabase.refreshToken.create.mockResolvedValue({
        id: "refresh_123",
        token: "mock_refresh_token",
        userId,
        userAgent: "",
        createdAt: new Date(),
      });

      const verifyResponse = await request
        .post("/auth/verify")
        .send({ code: newVerificationCode })
        .expect(200);

      expect(
        (verifyResponse.body as { needsOnboarding: boolean }).needsOnboarding,
      ).toBe(true);
    });
  });

  describe("Edge Case: Username Conflict During Onboarding", () => {
    it("should handle username taken by another user", async () => {
      const verificationCode = "clg0p72z4000008ld12345678";
      const userId = "user_123";

      // Setup: User completes verification
      const verificationToken = {
        code: verificationCode,
        email: testEmail,
        expiresAt: new Date(Date.now() + 15 * 60 * 1000),
        createdAt: new Date(),
      };

      const newUser = {
        id: userId,
        email: testEmail,
        username: `user_${Date.now()}`,
        verified: true,
        onboardingComplete: false,
        elo: 200,
        createdAt: new Date(),
        updatedAt: new Date(),
      };

      mockDatabase.verificationToken.create.mockResolvedValue(
        verificationToken,
      );
      mockDatabase.verificationToken.findFirst.mockResolvedValue(
        verificationToken,
      );
      mockDatabase.user.findUnique.mockResolvedValue(null);
      mockDatabase.user.create.mockResolvedValue(newUser);
      mockDatabase.verificationToken.delete.mockResolvedValue(
        verificationToken,
      );
      mockDatabase.refreshToken.create.mockResolvedValue({
        id: "refresh_123",
        token: "mock_refresh_token",
        userId,
        userAgent: "",
        createdAt: new Date(),
      });

      await request
        .post("/auth/magic-link")
        .send({ email: testEmail })
        .expect(200);

      const verifyResponse = await request
        .post("/auth/verify")
        .send({ code: verificationCode })
        .expect(200);

      // Attempt 1: Try username that's taken
      const userWithTakenUsername = {
        id: "different_user_id",
        email: "other@example.com",
        username: testUsername,
        verified: true,
        onboardingComplete: true,
        elo: 500,
        createdAt: new Date(),
        updatedAt: new Date(),
      };

      // requireUser middleware will fetch the user from database
      mockDatabase.user.findUnique.mockResolvedValueOnce(newUser); // requireUser middleware
      mockDatabase.user.findUnique.mockResolvedValueOnce(userWithTakenUsername); // Username check

      await request
        .post("/onboarding/complete")
        .set(
          "Authorization",
          `Bearer ${(verifyResponse.body as { accessToken: string }).accessToken}`,
        )
        .send({
          username: testUsername,
          skillLevel: testSkillLevel,
        })
        .expect(409);

      // Attempt 2: Try with available username
      const availableUsername = "available_username";
      const updatedUser = {
        ...newUser,
        username: availableUsername,
        elo: 500,
        onboardingComplete: true,
      };

      const expectedUserResponse = {
        ...updatedUser,
        createdAt: expect.any(String) as string,
        updatedAt: expect.any(String) as string,
      };

      // requireUser middleware will fetch the user from database
      mockDatabase.user.findUnique.mockResolvedValueOnce(newUser); // requireUser middleware
      mockDatabase.user.findUnique.mockResolvedValueOnce(null); // Username available
      mockDatabase.user.update.mockResolvedValue(updatedUser);

      const onboardingResponse = await request
        .post("/onboarding/complete")
        .set(
          "Authorization",
          `Bearer ${(verifyResponse.body as { accessToken: string }).accessToken}`,
        )
        .send({
          username: availableUsername,
          skillLevel: testSkillLevel,
        })
        .expect(200);

      expect(onboardingResponse.body).toEqual(expectedUserResponse);
    });
  });

  describe("Edge Case: Double Onboarding Attempt", () => {
    it("should prevent completing onboarding twice", async () => {
      const verificationCode = "clg0p72z4000008ld12345678";
      const userId = "user_123";

      // User already completed onboarding
      const completedUser = {
        id: userId,
        email: testEmail,
        username: testUsername,
        verified: true,
        onboardingComplete: true,
        elo: 500,
        createdAt: new Date(),
        updatedAt: new Date(),
      };

      // Login flow
      const verificationToken = {
        code: verificationCode,
        email: testEmail,
        expiresAt: new Date(Date.now() + 15 * 60 * 1000),
        createdAt: new Date(),
      };

      mockDatabase.verificationToken.create.mockResolvedValue(
        verificationToken,
      );
      mockDatabase.verificationToken.findFirst.mockResolvedValue(
        verificationToken,
      );
      mockDatabase.user.findUnique.mockResolvedValue(completedUser);
      mockDatabase.verificationToken.delete.mockResolvedValue(
        verificationToken,
      );
      mockDatabase.refreshToken.create.mockResolvedValue({
        id: "refresh_123",
        token: "mock_refresh_token",
        userId,
        userAgent: "",
        createdAt: new Date(),
      });

      await request
        .post("/auth/magic-link")
        .send({ email: testEmail })
        .expect(200);

      const verifyResponse = await request
        .post("/auth/verify")
        .send({ code: verificationCode })
        .expect(200);

      expect(
        (verifyResponse.body as { needsOnboarding: boolean }).needsOnboarding,
      ).toBe(false);

      // Attempt to complete onboarding again
      // requireUser middleware will fetch the user from database
      mockDatabase.user.findUnique.mockResolvedValue(completedUser); // requireUser middleware

      await request
        .post("/onboarding/complete")
        .set(
          "Authorization",
          `Bearer ${(verifyResponse.body as { accessToken: string }).accessToken}`,
        )
        .send({
          username: "new_username",
          skillLevel: "advanced",
        })
        .expect(403);
    });
  });

  describe("Error Cases", () => {
    it("should handle invalid email format", async () => {
      await request
        .post("/auth/magic-link")
        .send({ email: "invalid-email" })
        .expect(400);
    });

    it("should handle missing authorization header", async () => {
      await request
        .post("/onboarding/complete")
        .send({
          username: testUsername,
          skillLevel: testSkillLevel,
        })
        .expect(401);
    });

    it("should handle malformed authorization header", async () => {
      await request
        .post("/onboarding/complete")
        .set("Authorization", "InvalidToken")
        .send({
          username: testUsername,
          skillLevel: testSkillLevel,
        })
        .expect(401);
    });

    it("should handle invalid verification code format", async () => {
      await request
        .post("/auth/verify")
        .send({ code: "invalid_format" })
        .expect(400);
    });

    it("should handle nonexistent verification code", async () => {
      mockDatabase.verificationToken.findFirst.mockResolvedValue(null);

      await request
        .post("/auth/verify")
        .send({ code: "clg0p72z4000008ld11112222" })
        .expect(400);
    });
  });
});
