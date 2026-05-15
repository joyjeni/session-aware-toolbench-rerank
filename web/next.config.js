/** @type {import('next').NextConfig} */
module.exports = {
  reactStrictMode: true,
  // The /api/rerank route reads a JSON file synchronously
  experimental: { serverComponentsExternalPackages: [] },
};
