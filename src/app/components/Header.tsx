import Link from "next/link";
import SoccerBallIcon from "./SoccerBallIcon";

export default function Header() {
  return (
    <header className="sticky top-0 z-10 bg-white/80 shadow-sm backdrop-blur-sm">
      <div className="mx-auto max-w-7xl px-4 py-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between">
          <Link
            href="/"
            className="flex items-center space-x-2 transition-opacity hover:opacity-80"
          >
            <SoccerBallIcon className="h-8 w-8 text-green-600" size={32} />
            <h1 className="text-2xl font-bold text-gray-900">Taka</h1>
          </Link>
          <nav className="hidden items-center space-x-6 md:flex">
            <Link
              href="/rules"
              className="font-medium text-gray-600 transition-colors hover:text-gray-900"
            >
              Rules
            </Link>
            <Link
              href="#"
              className="font-medium text-gray-600 transition-colors hover:text-gray-900"
            >
              Tutorial
            </Link>
            <Link
              href="#"
              className="font-medium text-gray-600 transition-colors hover:text-gray-900"
            >
              About
            </Link>
          </nav>
          <button className="rounded-lg bg-green-600 px-6 py-2 font-medium text-white transition-colors hover:bg-green-700">
            Play Now
          </button>
        </div>
      </div>
    </header>
  );
}
