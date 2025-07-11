import type { ErrorHandler } from "@/types";

// This noinspection is here because TypeScript doesn't like the next() function being passed, but
// it is required on an express error handler
// eslint-disable-next-line @typescript-eslint/no-unused-vars
const errorHandler: ErrorHandler = (err, _req, res, _next) => {
  console.error("[500 INTERNAL SERVER ERROR]", err);
  res.status(500).send("Internal Server Error");
};

export default errorHandler;
