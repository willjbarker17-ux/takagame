import type { AuthenticatedRequest, RouteHandler } from "@/types";
import { verifyToken } from "@/utils/tokens";
import prisma from "@/database";

/**
 * Require a user to be authenticated and fetch their full profile
 */
const requireUser: RouteHandler = async (req, res, next) => {
  try {
    const authHeader = req.headers.authorization;

    if (!authHeader || !authHeader.startsWith("Bearer ")) {
      res.sendStatus(401);
      return;
    }

    // Get the token from the authorization header
    const token = authHeader.split(" ")[1];

    if (!token) {
      res.sendStatus(401);
      return;
    }

    const payload = verifyToken(token);

    // If the token is invalid, return a 403
    if (!payload) {
      res.sendStatus(403);
      return;
    }

    // Fetch the full user profile from the database
    const user = await prisma.user.findUnique({
      where: { id: payload.userId },
    });

    // If user doesn't exist in database, return 403
    if (!user) {
      res.sendStatus(403);
      return;
    }

    // Set the full user pr ofile on the request
    (req as AuthenticatedRequest).user = user;

    next();
  } catch (error) {
    return next(error);
  }
};

export default requireUser;
