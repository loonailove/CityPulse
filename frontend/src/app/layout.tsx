import { Analytics } from '@vercel/analytics/next'
import type { Metadata, Viewport } from 'next'
import { Space_Grotesk, Space_Mono } from 'next/font/google'
import './globals.css'

const grotesk = Space_Grotesk({
  subsets: ['latin', 'latin-ext'],
  weight: ['400', '500', '700'],
  variable: '--font-grotesk',
})

const mono = Space_Mono({
  subsets: ['latin', 'latin-ext'],
  weight: ['400', '700'],
  variable: '--font-mono-face',
})

export const metadata: Metadata = {
  title: 'CityPulse — One brand. Every city. Its own pulse.',
  description:
    "Upload your master ad and CityPulse adapts it to each city's moment, using weather, events and holidays.",
}

export const viewport: Viewport = {
  colorScheme: 'light',
  themeColor: '#faf5ee',
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en" className={`${grotesk.variable} ${mono.variable} bg-background`}>
      <body className="antialiased">
        {children}
        {process.env.NODE_ENV === 'production' && <Analytics />}
      </body>
    </html>
  )
}
