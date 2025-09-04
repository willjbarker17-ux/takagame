import { randomBytes } from "crypto";

export interface GuestSession {
  sessionId: string;
  username: string;
  createdAt: Date;
  expiresAt: Date;
  socketId?: string;
}

class GuestSessionManager {
  private sessions: Map<string, GuestSession> = new Map();
  private readonly SESSION_DURATION = 24 * 60 * 60 * 1000; // 24 hours

  generateSessionId(): string {
    return randomBytes(32).toString("hex");
  }

  generateGuestUsername(): string {
    const adjectives = [
      "Swift",
      "Bold",
      "Quick",
      "Smart",
      "Brave",
      "Cool",
      "Fast",
    ];
    const nouns = ["Player", "Gamer", "User", "Striker", "Champion", "Star"];
    const randomNum = Math.floor(Math.random() * 1000);

    const adjective = adjectives[Math.floor(Math.random() * adjectives.length)];
    const noun = nouns[Math.floor(Math.random() * nouns.length)];

    return `${adjective}${noun}${randomNum}`;
  }

  createSession(username?: string): GuestSession {
    const sessionId = this.generateSessionId();
    const guestUsername = username || this.generateGuestUsername();
    const now = new Date();
    const expiresAt = new Date(now.getTime() + this.SESSION_DURATION);

    const session: GuestSession = {
      sessionId,
      username: guestUsername,
      createdAt: now,
      expiresAt,
    };

    this.sessions.set(sessionId, session);
    this.scheduleCleanup(sessionId, this.SESSION_DURATION);

    return session;
  }

  getSession(sessionId: string): GuestSession | null {
    const session = this.sessions.get(sessionId);

    if (!session) {
      return null;
    }

    if (session.expiresAt < new Date()) {
      this.sessions.delete(sessionId);
      return null;
    }

    return session;
  }

  updateSocketId(sessionId: string, socketId: string): boolean {
    const session = this.sessions.get(sessionId);

    if (!session) {
      return false;
    }

    session.socketId = socketId;
    return true;
  }

  deleteSession(sessionId: string): boolean {
    return this.sessions.delete(sessionId);
  }

  private scheduleCleanup(sessionId: string, delay: number): void {
    setTimeout(() => {
      this.sessions.delete(sessionId);
    }, delay);
  }

  // Utility method for periodic cleanup of expired sessions
  cleanupExpiredSessions(): number {
    const now = new Date();
    let cleaned = 0;

    for (const [sessionId, session] of this.sessions) {
      if (session.expiresAt < now) {
        this.sessions.delete(sessionId);
        cleaned++;
      }
    }

    return cleaned;
  }

  getActiveSessionCount(): number {
    return this.sessions.size;
  }
}

export const guestSessionManager = new GuestSessionManager();

// Run cleanup every hour
setInterval(
  () => {
    const cleaned = guestSessionManager.cleanupExpiredSessions();
    if (cleaned > 0) {
      console.log(`Cleaned up ${cleaned} expired guest sessions`);
    }
  },
  60 * 60 * 1000,
);
