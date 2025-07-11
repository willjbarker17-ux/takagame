import dotenv from "dotenv";
import { z } from "zod";
import ms from "ms";

// Load base .env file first
dotenv.config();

// Then load environment-specific .env file if it exists
const nodeEnv = process.env["NODE_ENV"] || "development";
dotenv.config({ path: `.env.${nodeEnv}`, override: true });

// Zod schema for ms.StringValue, validates that the string is a valid ms format
const msStringSchema = z.string().refine(
  (val): val is ms.StringValue => {
    try {
      // Cast val to ms.StringValue for type safety
      return typeof ms(val as ms.StringValue) === "number";
    } catch {
      return false;
    }
  },
  { message: "Invalid ms string value" },
);

const envSchema = z.object({
  NODE_ENV: z
    .enum(["development", "production", "test"])
    .default("development"),
  PORT: z.coerce.number().default(3000),
  SITE_URL: z
    .string()
    .startsWith("http")
    .refine((url) => !url.endsWith("/")), // Frontend URL
  BACKEND_URL: z
    .string()
    .startsWith("http")
    .refine((url) => !url.endsWith("/")), // Backend API URL
  DATABASE_URL: z.string().url(),
  SENTRY_DSN: z.string().optional(), // Required in prod

  /* JWT config */
  JWT_SECRET: z.string(),
  // JWT_EXPIRES_IN and JWT_REFRESH_EXPIRES_IN must be valid ms string values
  JWT_EXPIRES_IN: msStringSchema.default("15m"),
  JWT_REFRESH_EXPIRES_IN: msStringSchema.default("7d"),
  JWT_REFRESH_TOKEN_COOKIE_NAME: z.string().default("token"),

  /* Email config */
  // Option to send emails to the console, instead of actually sending the email.
  EMAIL_STUB_ENABLED: z.coerce.boolean().default(false),

  GMAIL_USER: z.string().optional(), // Required in prod
  GMAIL_PASSWORD: z.string().optional(), // Required in prod
});

const parsedEnv = envSchema.safeParse(process.env);

if (!parsedEnv.success) {
  console.error(
    "Invalid environment variables:",
    parsedEnv.error.flatten().fieldErrors,
  );
  throw new Error("Invalid environment variables.");
}

// This allows us to add additional properties to the config object
const config = {
  ...parsedEnv.data,
  PRODUCTION: parsedEnv.data.NODE_ENV === "production",
};

// Production required variables
if (config.PRODUCTION) {
  if (config.EMAIL_STUB_ENABLED) {
    throw new Error("EMAIL_STUB_ENABLED is not allowed in production.");
  }

  if (!config.SENTRY_DSN) {
    throw new Error("SENTRY_DSN is required in production.");
  }

  if (!config.GMAIL_USER || !config.GMAIL_PASSWORD) {
    throw new Error(
      "GMAIL_USER and GMAIL_PASSWORD are required in production.",
    );
  }
}

export default config;
