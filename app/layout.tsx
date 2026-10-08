import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Atelier Concierge',
  description: 'Chat with the hotel concierge to plan and complete room requests.',
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
