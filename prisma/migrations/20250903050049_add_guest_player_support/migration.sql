-- DropForeignKey
ALTER TABLE "Game" DROP CONSTRAINT "Game_whitePlayerId_fkey";

-- AlterTable
ALTER TABLE "Game" ADD COLUMN     "blackPlayerGuestId" TEXT,
ADD COLUMN     "blackPlayerUsername" TEXT,
ADD COLUMN     "whitePlayerGuestId" TEXT,
ADD COLUMN     "whitePlayerUsername" TEXT,
ALTER COLUMN "whitePlayerId" DROP NOT NULL;

-- AddForeignKey
ALTER TABLE "Game" ADD CONSTRAINT "Game_whitePlayerId_fkey" FOREIGN KEY ("whitePlayerId") REFERENCES "User"("id") ON DELETE SET NULL ON UPDATE CASCADE;
