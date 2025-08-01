import { dirname } from "path";
import { fileURLToPath } from "url";
import { FlatCompat } from "@eslint/eslintrc";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const compat = new FlatCompat({
  baseDirectory: __dirname,
});

const eslintConfig = [
  ...compat.extends("next/core-web-vitals", "next/typescript"),
  ...compat.plugins("react", "react-hooks"),
  {
    rules: {
      // React specific rules
      "react/prop-types": "off", // We use TypeScript for prop validation
      "react/react-in-jsx-scope": "off", // Not needed in Next.js
      "react/jsx-uses-react": "off", // Not needed in Next.js

      // React Hooks rules
      "react-hooks/rules-of-hooks": "error",
      "react-hooks/exhaustive-deps": "warn",

      // Additional React best practices
      "react/jsx-key": "error",
      "react/jsx-no-duplicate-props": "error",
      "react/jsx-no-undef": "error",
      "react/jsx-uses-vars": "error",
      "react/no-deprecated": "warn",
      "react/no-direct-mutation-state": "error",
      "react/no-is-mounted": "error",
      "react/no-render-return-value": "error",
      "react/require-render-return": "error",
      "react/self-closing-comp": "warn",
      "react/jsx-fragments": ["warn", "syntax"],
    },
  },
];

export default eslintConfig;
