import { createAccessToken, createRefreshToken } from "@/utils/tokens";

export function createTestTokens(userId: string) {
  const accessToken = createAccessToken(userId);
  const refreshToken = createRefreshToken(userId);

  return { accessToken, refreshToken };
}

export function createAuthHeader(accessToken: string) {
  return { Authorization: `Bearer ${accessToken}` };
}
