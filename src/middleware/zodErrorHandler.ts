import type { ErrorHandler } from "@/types";
import { ZodError, type ZodIssue } from "zod";

const zodErrorHandler: ErrorHandler = (err, _req, res, next) => {
  if (err instanceof ZodError) {
    const errorMessages = err.errors.map((issue: ZodIssue) => ({
      message: `${issue.path.join(".") || "value"} is ${issue.message}`,
    }));

    res.status(400).json({
      error: "Invalid data",
      details: errorMessages,
    });
    return;
  }

  next(err);
};

export default zodErrorHandler;
