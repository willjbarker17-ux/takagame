"use client";

import { useState, useEffect, useCallback } from "react";
import type { GuestSession } from "@/services/socketService";

interface User {
  id: string;
  username: string;
  elo: number;
  verified: boolean;
  onboardingCompleted: boolean;
}

interface AuthState {
  user: User | null;
  accessToken: string | null;
  isAuthenticated: boolean;
  isGuest: boolean;
  guestSession: GuestSession | null;
  guestUsername: string | null;
  isLoading: boolean;
  error: string | null;
}

/**
 * Authentication hook for managing both guest and authenticated user sessions
 */
export const useAuth = () => {
  const [authState, setAuthState] = useState<AuthState>({
    user: null,
    accessToken: null,
    isAuthenticated: false,
    isGuest: false,
    guestSession: null,
    guestUsername: null,
    isLoading: true,
    error: null,
  });

  useEffect(() => {
    // Check for stored auth data on mount
    const storedToken = localStorage.getItem("taka_access_token");
    const storedUser = localStorage.getItem("taka_user");
    const storedGuestSession = localStorage.getItem("taka_guest_session");
    const storedGuestUsername = localStorage.getItem("taka_guest_username");

    // Check for authenticated user first
    if (storedToken && storedUser) {
      try {
        const user = JSON.parse(storedUser);
        setAuthState({
          user,
          accessToken: storedToken,
          isAuthenticated: true,
          isGuest: false,
          guestSession: null,
          guestUsername: null,
          isLoading: false,
          error: null,
        });
        return;
      } catch {
        // Invalid stored data, clear it
        localStorage.removeItem("taka_access_token");
        localStorage.removeItem("taka_user");
      }
    }

    // Check for guest session
    if (storedGuestSession && storedGuestUsername) {
      try {
        console.log("Parsing stored guest session:", storedGuestSession);
        const guestSession = JSON.parse(storedGuestSession);
        console.log("Parsed guest session successfully:", guestSession);
        setAuthState({
          user: null,
          accessToken: null,
          isAuthenticated: false,
          isGuest: true,
          guestSession,
          guestUsername: storedGuestUsername,
          isLoading: false,
          error: null,
        });
        console.log("Guest session restored to auth state");
        return;
      } catch (error) {
        console.error("Failed to parse stored guest session:", error);
        // Invalid stored data, clear it
        localStorage.removeItem("taka_guest_session");
        localStorage.removeItem("taka_guest_username");
      }
    }

    // Check for guest username only (session will be created later)
    // Only do this if we don't have a stored session
    if (storedGuestUsername && !storedGuestSession) {
      console.log("Restoring guest username only (no session found)");
      setAuthState({
        user: null,
        accessToken: null,
        isAuthenticated: false,
        isGuest: true,
        guestSession: null,
        guestUsername: storedGuestUsername,
        isLoading: false,
        error: null,
      });
      return;
    }

    // No valid auth data found - this is a completely new user
    console.log("No stored auth data found - new user");
    setAuthState((prev) => ({ ...prev, isLoading: false }));
  }, []);

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const login = async (email: string, password: string) => {
    setAuthState((prev) => ({ ...prev, isLoading: true, error: null }));

    try {
      // TODO: Replace with actual API call when auth backend is ready
      // For now, use mock authentication
      const mockUser: User = {
        id: "user_" + Date.now(),
        username: "Player" + Math.floor(Math.random() * 1000),
        elo: 1200 + Math.floor(Math.random() * 600),
        verified: true,
        onboardingCompleted: true,
      };

      const mockToken = "mock_jwt_token_" + Date.now();

      // Clear any guest data
      localStorage.removeItem("taka_guest_session");
      localStorage.removeItem("taka_guest_username");

      // Store auth data
      localStorage.setItem("taka_access_token", mockToken);
      localStorage.setItem("taka_user", JSON.stringify(mockUser));

      setAuthState({
        user: mockUser,
        accessToken: mockToken,
        isAuthenticated: true,
        isGuest: false,
        guestSession: null,
        guestUsername: null,
        isLoading: false,
        error: null,
      });

      return { success: true };
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : "Login failed";
      setAuthState((prev) => ({
        ...prev,
        isLoading: false,
        error: errorMessage,
      }));
      return { success: false, error: errorMessage };
    }
  };

  const loginAsGuest = (username?: string) => {
    setAuthState((prev) => ({ ...prev, isLoading: true, error: null }));

    try {
      // If we already have a guest session, don't clear it unless explicitly creating a new one
      const currentGuestSession = authState.guestSession;
      const guestUsername =
        username ||
        authState.guestUsername ||
        `Player${Math.floor(Math.random() * 10000)}`;

      // Clear any authenticated user data
      localStorage.removeItem("taka_access_token");
      localStorage.removeItem("taka_user");

      // Store guest username (session will be created when creating game)
      localStorage.setItem("taka_guest_username", guestUsername);

      setAuthState({
        user: null,
        accessToken: null,
        isAuthenticated: false,
        isGuest: true,
        guestSession: currentGuestSession, // Preserve existing session if available
        guestUsername,
        isLoading: false,
        error: null,
      });

      return { success: true, username: guestUsername };
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : "Guest login failed";
      setAuthState((prev) => ({
        ...prev,
        isLoading: false,
        error: errorMessage,
      }));
      return { success: false, error: errorMessage };
    }
  };

  const setGuestSession = (session: GuestSession) => {
    localStorage.setItem("taka_guest_session", JSON.stringify(session));
    localStorage.setItem("taka_guest_username", session.username);

    setAuthState((prev) => ({
      ...prev,
      guestSession: session,
      guestUsername: session.username,
    }));
  };

  const logout = () => {
    // Clear both authenticated and guest data
    localStorage.removeItem("taka_access_token");
    localStorage.removeItem("taka_user");
    localStorage.removeItem("taka_guest_session");
    localStorage.removeItem("taka_guest_username");

    setAuthState({
      user: null,
      accessToken: null,
      isAuthenticated: false,
      isGuest: false,
      guestSession: null,
      guestUsername: null,
      isLoading: false,
      error: null,
    });
  };

  const clearError = () => {
    setAuthState((prev) => ({ ...prev, error: null }));
  };

  // Helper to check if user has any valid authentication (guest or authenticated)
  const hasValidAuth = useCallback(() => {
    return authState.isAuthenticated || authState.isGuest;
  }, [authState.isAuthenticated, authState.isGuest]);

  const stableLoginAsGuest = useCallback(loginAsGuest, [
    authState.guestSession,
    authState.guestUsername,
  ]);
  const stableSetGuestSession = useCallback(setGuestSession, []);

  return {
    ...authState,
    login,
    loginAsGuest: stableLoginAsGuest,
    logout,
    setGuestSession: stableSetGuestSession,
    clearError,
    hasValidAuth,
  };
};
