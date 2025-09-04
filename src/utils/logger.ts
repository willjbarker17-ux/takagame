type LogLevel = "info" | "warn" | "error" | "debug";

interface LogContext {
  socketId?: string;
  userId?: string;
  ip?: string;
  userAgent?: string;
  metadata?: Record<string, unknown>;
}

class Logger {
  private log(
    level: LogLevel,
    message: string,
    error?: Error,
    context?: LogContext,
  ) {
    const timestamp = new Date().toISOString();
    const logEntry = {
      timestamp,
      level,
      message,
      ...(error && {
        error: { name: error.name, message: error.message, stack: error.stack },
      }),
      ...(context && { context }),
    };

    console.log(JSON.stringify(logEntry, null, 2));
  }

  info(message: string, context?: LogContext) {
    this.log("info", message, undefined, context);
  }

  warn(message: string, context?: LogContext) {
    this.log("warn", message, undefined, context);
  }

  error(message: string, error: Error, context?: LogContext) {
    this.log("error", message, error, context);
  }

  debug(message: string, context?: LogContext) {
    this.log("debug", message, undefined, context);
  }

  socket(message: string, context?: LogContext) {
    this.log("info", `[SOCKET] ${message}`, undefined, context);
  }
}

export const logger = new Logger();
