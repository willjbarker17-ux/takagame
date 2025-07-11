import { vi } from "vitest";
import mockDatabase from "@tests/mocks/database";

vi.mock("@/database", () => ({ default: mockDatabase }));

vi.mock("@/utils/email", () => ({
  sendMagicLinkEmail: vi.fn().mockResolvedValue(undefined),
}));

vi.spyOn(console, "warn").mockImplementation(() => {});
vi.spyOn(console, "error").mockImplementation(() => {});
