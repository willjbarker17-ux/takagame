import { vi } from "vitest";
import mockPrisma from "@tests/mocks/prisma";

vi.mock("@/database", () => ({ default: mockPrisma }));

vi.mock("@/utils/email", () => ({
  sendMagicLinkEmail: vi.fn().mockResolvedValue(undefined),
}));
