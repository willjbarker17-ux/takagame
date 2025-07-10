import nodemailer from "nodemailer";
import config from "@/config";

// Create a reusable transporter object using SMTP transport
const transporter = nodemailer.createTransport({
  host: "smtp.gmail.com",
  port: 587,
  secure: false,
  auth: {
    user: config.GMAIL_USER,
    pass: config.GMAIL_PASSWORD,
  },
});

/**
 * Sends an email using nodemailer and SMTP configuration.
 * If EMAIL_STUB_ENABLED is true, logs the email to the console instead of sending.
 * @param options Email sending options
 */
async function sendEmail(
  to: string,
  subject: string,
  html: string,
  from?: string,
): Promise<void> {
  if (config.EMAIL_STUB_ENABLED) {
    // Log email details to the console for development/testing
    console.log("[EMAIL STUB] To:", to);
    console.log("[EMAIL STUB] Subject:", subject);
    console.log("[EMAIL STUB] HTML:", html);
    return;
  }

  // Send the email using the transporter
  await transporter.sendMail({
    from,
    to,
    subject,
    html,
  });
}

export function sendMagicLinkEmail(to: string, token: string) {
  const url = `${config.SITE_URL}/magic-link?token=${token}`;
  const html = `
    <p>Hello!</p>
    <p>Click <a href="${url}">here</a> to login to your account.</p>
    <p>This link will expire in 15 minutes.</p>
    <p>If you did not request this login link, please ignore this email.</p>
  `;
  return sendEmail(to, "Magic Link", html);
}
