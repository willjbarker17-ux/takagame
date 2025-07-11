import { beforeEach } from "vitest";
import { mockDeep, mockReset } from "vitest-mock-extended";
import { PrismaClient } from "@prisma/client";

const prisma = mockDeep<PrismaClient>();

beforeEach(() => {
  mockReset(prisma);
});

export default prisma;
