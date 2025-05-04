/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./index.html",
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: "#1E3A8A", // Headers, nav bar, primary buttons
        secondary: "#3B82F6", // Secondary buttons, hover effects
        accent: "#10B981", // Success messages, chart highlights
        neutralLight: "#F9FAFB", // Backgrounds, table cells
        neutralDark: "#1F2937", // Text, table borders
      },
    },
  },
  plugins: [],
};
  