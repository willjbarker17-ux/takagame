import type { RouteHandler } from "@/types";
import { verifyToken } from "@/utils/tokens";

/**
 * Require a user to be authenticated
 */
const requireUser: RouteHandler = (req, res, next) => {
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

  // Set the user payload on the request
  req.user = payload;

  next();
};

export default requireUser;
