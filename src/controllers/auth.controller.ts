import { z } from "zod";
import type { RouteHandler } from "../types";
import prisma from "@/database";
import config from "../config";
import { sendMagicLinkEmail } from "../utils/email";
import {
  createAccessToken,
  createRefreshToken,
  verifyToken,
} from "../utils/tokens";
import ms from "ms";
import type { CookieOptions } from "express";

const clearRefreshTokenCookieConfig: CookieOptions = {
  httpOnly: true,
  sameSite: "lax",
  secure: config.PRODUCTION,
};

const setRefreshTokenCookieConfig: CookieOptions = {
  ...clearRefreshTokenCookieConfig,
  maxAge: ms(config.JWT_REFRESH_EXPIRES_IN),
};

const emailRequestOtpSchema = z
  .object({
    email: z.string().email("Email is invalid"),
  })
  .strict();

/**
 * Request a magic link for an email to sign in or sign up. This route should be called from the frontend.
 * Works for both new users (signup) and existing users (login).
 */
export const emailRequestOtp: RouteHandler = async (req, res, next) => {
  try {
    const { email } = emailRequestOtpSchema.parse(req.body);

    // Create verification token
    const expiresAt = new Date();
    expiresAt.setMinutes(expiresAt.getMinutes() + 15); // Expires in 15 minutes

    const { code } = await prisma.verificationToken.create({
      data: {
        email,
        expiresAt,
      },
    });

    await sendMagicLinkEmail(email, code);

    res.sendStatus(200);
  } catch (error) {
    return next(error);
  }
};

const emailVerifyOtpSchema = z
  .object({
    code: z.string().cuid("Code is invalid"),
  })
  .strict();

/**
 * Verify the OTP after sending it to email. This route should be called from the frontend.
 * If user exists and has completed onboarding, they get logged in.
 * If user exists but hasn't completed onboarding, they get a token but need to complete onboarding.
 */
export const emailVerifyOtp: RouteHandler = async (req, res, next) => {
  try {
    const { code } = emailVerifyOtpSchema.parse(req.body);

    const token = await prisma.verificationToken.findFirst({
      where: {
        code,
      },
    });

    if (!token) {
      res.status(400).json({ error: "Verification request not found" });
      return;
    }

    if (token.expiresAt < new Date()) {
      res.status(400).json({ error: "Login request has expired" });
      return;
    }

    let user = await prisma.user.findUnique({
      where: {
        email: token.email,
      },
    });

    // If no user exists, create one but don't complete onboarding yet
    if (!user) {
      user = await prisma.user.create({
        data: {
          email: token.email,
          username: `user_${Date.now()}`, // Temporary username, will be updated during onboarding
          verified: true,
          onboardingComplete: false,
        },
      });
    } else {
      // If user exists but is not verified, verify them
      if (!user.verified) {
        user = await prisma.user.update({
          where: { id: user.id },
          data: { verified: true },
        });
      }
    }

    await prisma.verificationToken.delete({
      where: {
        code,
      },
    });

    // Check if an existing refresh token cookie needs to be invalidated:
    // - Delete single token if it belongs to current user
    // - Delete all user's tokens if token is invalid/belongs to different user
    // - Clear the cookie in both cases
    if (req.cookies?.[config.JWT_REFRESH_TOKEN_COOKIE_NAME]) {
      const refreshToken = req.cookies[config.JWT_REFRESH_TOKEN_COOKIE_NAME] as
        | string
        | undefined;

      const foundRefreshToken = await prisma.refreshToken.findUnique({
        where: {
          token: refreshToken,
        },
      });

      // Delete all refresh tokens if token is invalid or belongs to another user
      if (!foundRefreshToken || foundRefreshToken.userId !== user.id) {
        await prisma.refreshToken.deleteMany({
          where: {
            userId: user.id,
          },
        });
      } else {
        // Delete the single token if it belongs to the current user
        await prisma.refreshToken.delete({
          where: {
            token: refreshToken,
          },
        });
      }

      res.clearCookie(
        config.JWT_REFRESH_TOKEN_COOKIE_NAME,
        clearRefreshTokenCookieConfig,
      );
    }

    const accessToken = createAccessToken(user.id);
    const refreshToken = createRefreshToken(user.id);

    await prisma.refreshToken.create({
      data: {
        token: refreshToken,
        userId: user.id,
        userAgent: req.headers["user-agent"] ?? "",
      },
    });

    // Set the new refresh token as an HttpOnly cookie
    res.cookie(
      config.JWT_REFRESH_TOKEN_COOKIE_NAME,
      refreshToken,
      setRefreshTokenCookieConfig,
    );

    res.json({
      accessToken,
      needsOnboarding: !user.onboardingComplete,
    });
  } catch (error) {
    return next(error);
  }
};

const logoutSchema = z
  .object({
    refreshToken: z.string().min(1),
  })
  .strict();

/**
 * Log out a user
 */
export const logout: RouteHandler = async (req, res, next) => {
  try {
    const { refreshToken } = logoutSchema.parse(req.body);

    await prisma.refreshToken.delete({
      where: {
        token: refreshToken,
      },
    });

    res.sendStatus(200);
  } catch (error) {
    return next(error);
  }
};

const refreshTokenSchema = z
  .object({
    refreshToken: z.string().min(1),
  })
  .strict();

/**
 * Refresh a user's access token
 */
export const refreshToken: RouteHandler = async (req, res, next) => {
  try {
    const { refreshToken } = refreshTokenSchema.parse(req.body);

    const foundRefreshToken = await prisma.refreshToken.findUnique({
      where: {
        token: refreshToken,
      },
    });

    if (!foundRefreshToken) {
      const payload = verifyToken(refreshToken);

      if (payload) {
        // If the JWT was valid but not in the database, it's a possible stolen token, revoke all tokens
        await prisma.refreshToken.deleteMany({
          where: {
            userId: payload.userId,
          },
        });
      }

      res.status(401).json({ error: "Invalid refresh token" });
      return;
    }

    const payload = verifyToken(foundRefreshToken.token);

    if (!payload || payload.userId !== foundRefreshToken.userId) {
      res.status(401).json({ error: "Invalid refresh token" });
      return;
    }

    const accessToken = createAccessToken(payload.userId);

    if (!accessToken) {
      res.status(401).json({ error: "Invalid refresh token" });
      return;
    }

    res.json({ accessToken });
  } catch (error) {
    return next(error);
  }
};
