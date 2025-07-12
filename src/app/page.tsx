import StaticGameBoard from "./components/StaticGameBoard";
import SoccerBallIcon from "./components/SoccerBallIcon";

export default function Home() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 to-blue-50">
      {/* Hero Section */}
      <section className="mx-auto max-w-7xl px-4 py-20 sm:px-6 lg:px-8">
        <div className="text-center">
          <h2 className="animate-fade-in-up mb-6 text-5xl font-bold text-gray-900 opacity-0 sm:text-6xl">
            The Ultimate
            <span className="block text-green-600">Football Strategy Game</span>
          </h2>
          <p className="animate-fade-in-up animate-delay-200 mx-auto mb-8 max-w-3xl text-xl text-gray-600 opacity-0">
            Experience the tactical depth of football in a turn-based board
            game. Plan your moves, execute perfect passes, and outmaneuver your
            opponent on a 10x14 grid battlefield.
          </p>
          <div className="animate-fade-in-up animate-delay-400 flex flex-col justify-center gap-4 opacity-0 sm:flex-row">
            <button className="rounded-lg bg-green-600 px-8 py-4 text-lg font-semibold text-white transition-colors hover:bg-green-700">
              Start Playing
            </button>
            <a
              href="/rules"
              className="rounded-lg border border-gray-300 bg-white px-8 py-4 text-lg font-semibold text-gray-900 transition-colors hover:bg-gray-50"
            >
              Learn Rules
            </a>
          </div>
        </div>

        {/* Game Board */}
        <div className="animate-fade-in-up animate-delay-600 mt-16 opacity-0">
          <StaticGameBoard />
        </div>
      </section>

      {/* Features Section */}
      <section className="animate-fade-in-up animate-delay-800 bg-white py-20 opacity-0">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="animate-fade-in-up animate-delay-1000 mb-16 text-center opacity-0">
            <h3 className="mb-4 text-3xl font-bold text-gray-900">
              Why Choose Taka?
            </h3>
            <p className="text-xl text-gray-600">
              Strategic depth meets accessible gameplay
            </p>
          </div>

          <div className="grid gap-8 md:grid-cols-3">
            <div className="animate-fade-in-up animate-delay-1200 text-center opacity-0">
              <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-green-100">
                <span className="text-2xl">🎯</span>
              </div>
              <h4 className="mb-2 text-xl font-semibold text-gray-900">
                Strategic Gameplay
              </h4>
              <p className="text-gray-600">
                Every move matters. Plan your attacks, defend your goal, and
                execute perfect passes in this turn-based tactical experience.
              </p>
            </div>

            <div className="animate-fade-in-up animate-delay-1400 text-center opacity-0">
              <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-blue-100">
                <span className="text-2xl">👥</span>
              </div>
              <h4 className="mb-2 text-xl font-semibold text-gray-900">
                Local Multiplayer
              </h4>
              <p className="text-gray-600">
                Play with friends in pass-and-play mode. Take turns controlling
                your team of 11 pieces on the same device.
              </p>
            </div>

            <div className="animate-fade-in-up animate-delay-1600 text-center opacity-0">
              <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-purple-100">
                <span className="text-2xl">📱</span>
              </div>
              <h4 className="mb-2 text-xl font-semibold text-gray-900">
                Play Anywhere
              </h4>
              <p className="text-gray-600">
                Web-based game that works on any device. No downloads required -
                just open your browser and start playing.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* How to Play Section */}
      <section className="animate-fade-in-up animate-delay-1800 bg-gray-50 py-20 opacity-0">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="animate-fade-in-up animate-delay-2000 mb-16 text-center opacity-0">
            <h3 className="mb-4 text-3xl font-bold text-gray-900">
              How to Play
            </h3>
            <p className="text-xl text-gray-600">
              Master the basics in minutes, perfect your strategy over time
            </p>
          </div>

          <div className="grid gap-8 md:grid-cols-2 lg:grid-cols-4">
            <div className="animate-fade-in-up animate-delay-2200 text-center opacity-0">
              <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-green-600 text-lg font-bold text-white">
                1
              </div>
              <h4 className="mb-2 text-lg font-semibold text-gray-900">
                Choose Your Team
              </h4>
              <p className="text-gray-600">
                Start with 11 pieces including 1 goalie. White team moves first.
              </p>
            </div>

            <div className="animate-fade-in-up animate-delay-2400 text-center opacity-0">
              <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-green-600 text-lg font-bold text-white">
                2
              </div>
              <h4 className="mb-2 text-lg font-semibold text-gray-900">
                Move & Pass
              </h4>
              <p className="text-gray-600">
                Move pieces strategically, pass the ball, and control the field.
              </p>
            </div>

            <div className="animate-fade-in-up animate-delay-2600 text-center opacity-0">
              <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-green-600 text-lg font-bold text-white">
                3
              </div>
              <h4 className="mb-2 text-lg font-semibold text-gray-900">
                Score Goals
              </h4>
              <p className="text-gray-600">
                Get into the shooting zone and aim for the opponent&apos;s goal.
              </p>
            </div>

            <div className="animate-fade-in-up animate-delay-2800 text-center opacity-0">
              <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-green-600 text-lg font-bold text-white">
                4
              </div>
              <h4 className="mb-2 text-lg font-semibold text-gray-900">
                Win the Match
              </h4>
              <p className="text-gray-600">
                Outscore your opponent and claim victory on the tactical field.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="animate-fade-in-up animate-delay-3000 bg-green-600 py-20 opacity-0">
        <div className="mx-auto max-w-7xl px-4 text-center sm:px-6 lg:px-8">
          <h3 className="animate-fade-in-up animate-delay-3200 mb-4 text-3xl font-bold text-white opacity-0">
            Ready to Play?
          </h3>
          <p className="animate-fade-in-up animate-delay-3400 mb-8 text-xl text-green-100 opacity-0">
            Challenge a friend and experience the thrill of tactical football
          </p>
          <div className="animate-fade-in-up animate-delay-3600 opacity-0">
            <button className="rounded-lg bg-white px-8 py-4 text-lg font-semibold text-green-600 transition-colors hover:bg-gray-100">
              Start Your First Game
            </button>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="animate-fade-in-up animate-delay-3800 bg-gray-900 py-12 text-white opacity-0">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="grid gap-8 md:grid-cols-4">
            <div className="animate-fade-in-up animate-delay-4000 opacity-0">
              <div className="mb-4 flex items-center space-x-2">
                <SoccerBallIcon className="h-8 w-8 text-green-600" size={32} />
                <span className="text-xl font-bold">Taka</span>
              </div>
              <p className="text-gray-400">
                The digital adaptation of the ultimate football strategy board
                game.
              </p>
            </div>

            <div className="animate-fade-in-up animate-delay-4200 opacity-0">
              <h4 className="mb-3 font-semibold">Game</h4>
              <ul className="space-y-2 text-gray-400">
                <li>
                  <a href="#" className="transition-colors hover:text-white">
                    Play Now
                  </a>
                </li>
                <li>
                  <a
                    href="/rules"
                    className="transition-colors hover:text-white"
                  >
                    Rules
                  </a>
                </li>
                <li>
                  <a href="#" className="transition-colors hover:text-white">
                    Tutorial
                  </a>
                </li>
              </ul>
            </div>

            <div className="animate-fade-in-up animate-delay-4400 opacity-0">
              <h4 className="mb-3 font-semibold">About</h4>
              <ul className="space-y-2 text-gray-400">
                <li>
                  <a
                    href="/rules"
                    className="transition-colors hover:text-white"
                  >
                    How to Play
                  </a>
                </li>
                <li>
                  <a href="#" className="transition-colors hover:text-white">
                    Strategy Guide
                  </a>
                </li>
                <li>
                  <a href="#" className="transition-colors hover:text-white">
                    FAQ
                  </a>
                </li>
              </ul>
            </div>

            <div className="animate-fade-in-up animate-delay-4600 opacity-0">
              <h4 className="mb-3 font-semibold">Connect</h4>
              <ul className="space-y-2 text-gray-400">
                <li>
                  <a href="#" className="transition-colors hover:text-white">
                    GitHub
                  </a>
                </li>
                <li>
                  <a href="#" className="transition-colors hover:text-white">
                    Discord
                  </a>
                </li>
                <li>
                  <a href="#" className="transition-colors hover:text-white">
                    Twitter
                  </a>
                </li>
              </ul>
            </div>
          </div>

          <div className="animate-fade-in-up animate-delay-4800 mt-8 border-t border-gray-800 pt-8 text-center text-gray-400 opacity-0">
            <p>&copy; 2024 Taka Digital Edition. All rights reserved.</p>
          </div>
        </div>
      </footer>
    </div>
  );
}
