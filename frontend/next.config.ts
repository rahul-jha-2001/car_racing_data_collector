import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */
  reactStrictMode: true,
  allowedDevOrigins: [
    "http://192.168.29.247:3000", // or whatever IP you're accessing from
  ],
  
};

export default nextConfig;
