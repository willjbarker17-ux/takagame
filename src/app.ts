import config from "@/config";
import routes from "@/routes";
import cors from "cors";
import errorHandler from "@/middleware/errorHandler";
import express from "express";
import * as Sentry from "@sentry/node";
import helmet from "helmet";
import logger from "morgan";
import cookieParser from "cookie-parser";
import zodErrorHandler from "@/middleware/zodErrorHandler";
import noCache from "@/middleware/noCache";

const app = express();

if (config.PRODUCTION) {
  Sentry.init({
    dsn: config.SENTRY_DSN,
  });
}

app.use(helmet());

app.use(noCache);

app.use(logger("dev"));

app.use(
  cors({
    origin: [config.SITE_URL, "http://localhost:1420", "http://localhost:8000"],
    methods: ["GET", "POST", "OPTIONS"],
    credentials: true,
  }),
);

// Parse JSON bodies for everything besides the stripe webhook
app.use((req, res, next) => {
  if (req.originalUrl === "/webhooks/stripe") {
    next();
  } else {
    express.json()(req, res, next);
  }
});

app.use(express.urlencoded({ extended: true }));

app.use(cookieParser());

routes(app);

// Error handlers
app.use(zodErrorHandler);
if (config.PRODUCTION) Sentry.setupExpressErrorHandler(app);
app.use(errorHandler);

app.listen(config.PORT, () => {
  console.log(`Listening on port ${config.PORT}`);
});
