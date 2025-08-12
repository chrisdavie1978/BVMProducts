import type { NextConfig } from "next";

const backendDomain = process.env.BACKEND_DOMAIN || "http://localhost:8000/";

const nextConfig: NextConfig = {
  /* config options here */
  async rewrites() {
    return [
      {
        source: "/api/chat",
        destination: backendDomain,
      },
    ];
  },
  output: "standalone",
};

module.exports = nextConfig;
