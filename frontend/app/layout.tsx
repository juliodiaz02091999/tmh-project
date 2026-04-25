import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "TMH Analyzer",
  description: "Upload or capture an image to estimate TMH.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="min-h-dvh bg-black text-white antialiased">{children}</body>
    </html>
  );
}

