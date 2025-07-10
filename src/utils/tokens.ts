import jwt from "jsonwebtoken";
import config from "@/config";

/**
 * This functions generates a valid access token
 *
 * @param userId - The user id of the user that owns this jwt
 * @returns Returns a valid access token
 */
export const createAccessToken = (userId: string): string => {
  return jwt.sign({ userId: userId }, config.JWT_SECRET, {
    expiresIn: config.JWT_EXPIRES_IN,
  });
};

/**
 * This functions generates a valid refresh token
 *
 * @param userId - The user id of the user that owns this jwt
 * @returns Returns a valid refresh token
 */
export const createRefreshToken = (userId: string): string => {
  return jwt.sign({ userId }, config.JWT_SECRET, {
    expiresIn: config.JWT_REFRESH_EXPIRES_IN,
  });
};

export type Payload = {
  userId: string;
  iat?: number;
  exp?: number;
};

/**
 * Verify a token
 *
 * @param token - The token to verify
 * @returns Returns the decoded token
 */
export const verifyToken = (token: string): Payload | null => {
  try {
    return jwt.verify(token, config.JWT_SECRET) as Payload;
  } catch (error) {
    console.error(error);
    return null;
  }
};
