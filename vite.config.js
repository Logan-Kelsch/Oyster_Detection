import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    watch: {
      ignored: [
        // Always use recursive glob patterns to match everything under these folders
        "**/ModelResearch/**",
        "**/ForHamid/**",
      ]
    }
  },
  base: "/yolov8-tfjs/",
  build: {
    chunkSizeWarningLimit: 2000
  }
});
