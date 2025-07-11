import express from "express";
import cors from "cors";
import cookieParser from "cookie-parser";
import routes from "@/routes";
import zodErrorHandler from "@/middleware/zodErrorHandler";
import errorHandler from "@/middleware/errorHandler";
import config from "@/config";

export function createTestApp() {
  const app = express();

  app.use(
    cors({
      origin: [
        config.SITE_URL,
        "http://localhost:1420",
        "http://localhost:8000",
      ],
      methods: ["GET", "POST", "OPTIONS"],
      credentials: true,
    }),
  );

  app.use(express.json());
  app.use(express.urlencoded({ extended: true }));
  app.use(cookieParser());

  routes(app);

  // Error handlers
  app.use(zodErrorHandler);
  app.use(errorHandler);

  return app;
}
