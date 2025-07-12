import React from "react";
import Link from "next/link";
import { ArrowUp, ArrowDown, ArrowLeft, ArrowRight } from "lucide-react";

const RulesPage: React.FC = () => {
  return (
    <div className="min-h-screen bg-gray-50">
      <div className="mx-auto max-w-4xl px-4 py-8 sm:px-6 lg:px-8">
        <div className="rounded-lg bg-white p-8 shadow-lg">
          <h1 className="mb-8 text-center text-4xl font-bold text-gray-900">
            Taka: Game Rules
          </h1>

          {/* Game Overview */}
          <section className="mb-12">
            <h2 className="mb-4 text-2xl font-bold text-gray-800">
              1. Game Overview
            </h2>
            <p className="mb-4 text-gray-600">
              Taka is a turn-based strategy board game simulating football
              (soccer). Two players control teams of 11 pieces on a grid, with
              the objective of scoring goals by moving the ball into the
              opponent&apos;s goal area.
            </p>
          </section>

          {/* Game Setup */}
          <section className="mb-12">
            <h2 className="mb-4 text-2xl font-bold text-gray-800">
              2. Game Setup
            </h2>

            <div className="mb-6 rounded-lg bg-green-50 p-6">
              <h3 className="mb-3 text-lg font-semibold">Board Layout</h3>
              <div className="mb-6">
                <p className="mb-2 text-sm text-gray-600">
                  • Board: 10x14 grid (Rows A-J, Columns 1-14)
                </p>
                <p className="mb-2 text-sm text-gray-600">
                  • White&apos;s Goal: Squares D1, E1, F1, G1
                </p>
                <p className="mb-2 text-sm text-gray-600">
                  • Black&apos;s Goal: Squares D14, E14, F14, G14
                </p>
              </div>
              <div className="rounded border bg-white p-4">
                <div className="mb-4 text-center text-sm font-medium">
                  Board Visualization
                </div>
                <div className="flex justify-center">
                  <div className="inline-block">
                    {/* Column indicators */}
                    <div className="mb-1 flex">
                      <div className="w-8"></div> {/* Space for row labels */}
                      {Array.from({ length: 14 }, (_, i) => (
                        <div
                          key={i}
                          className="flex aspect-square w-8 items-center justify-center text-xs font-medium text-gray-600"
                        >
                          {i + 1}
                        </div>
                      ))}
                    </div>

                    {/* Board with row indicators */}
                    <div className="flex">
                      {/* Row indicators */}
                      <div className="mr-1 flex flex-col">
                        {Array.from({ length: 10 }, (_, i) => (
                          <div
                            key={i}
                            className="flex aspect-square w-8 items-center justify-center text-xs font-medium text-gray-600"
                          >
                            {String.fromCharCode(65 + i)}
                          </div>
                        ))}
                      </div>

                      {/* Game board */}
                      <div className="grid grid-cols-14 grid-rows-10 gap-[1px] rounded bg-green-700 p-2 shadow-lg">
                        {/* Row A (0) */}
                        <div className="aspect-square w-8 bg-green-600"></div>
                        <div className="aspect-square w-8 bg-green-600"></div>
                        <div className="aspect-square w-8 bg-green-600"></div>
                        <div className="aspect-square w-8 bg-green-600"></div>
                        <div className="aspect-square w-8 border-r-4 border-r-white bg-green-600"></div>
                        <div className="aspect-square w-8 bg-green-600"></div>
                        <div className="aspect-square w-8 bg-green-600"></div>
                        <div className="aspect-square w-8 bg-green-600"></div>
                        <div className="aspect-square w-8 border-r-4 border-r-white bg-green-600"></div>
                        <div className="aspect-square w-8 bg-green-600"></div>
                        <div className="aspect-square w-8 bg-green-600"></div>
                        <div className="aspect-square w-8 bg-green-600"></div>
                        <div className="aspect-square w-8 bg-green-600"></div>
                        <div className="aspect-square w-8 bg-green-600"></div>

                        {/* Row B (1) - B6 */}
                        <div className="aspect-square w-8 bg-green-600"></div>
                        <div className="aspect-square w-8 bg-green-600"></div>
                        <div className="aspect-square w-8 bg-green-600"></div>
                        <div className="aspect-square w-8 bg-green-600"></div>
                        <div className="aspect-square w-8 border-r-4 border-r-white bg-green-600"></div>
                        <div className="flex aspect-square w-8 items-center justify-center bg-green-600">
                          <div className="h-4 w-4 rounded-full border-2 border-gray-400 bg-white shadow-lg"></div>
                        </div>
                        <div className="aspect-square w-8 bg-green-600"></div>
                        <div className="aspect-square w-8 bg-green-600"></div>
                        <div className="flex aspect-square w-8 items-center justify-center border-r-4 border-r-white bg-green-600">
                          <div className="h-4 w-4 rounded-full border-2 border-gray-600 bg-gray-800 shadow-lg"></div>
                        </div>
                        <div className="aspect-square w-8 bg-green-600"></div>
                        <div className="aspect-square w-8 bg-green-600"></div>
                        <div className="aspect-square w-8 bg-green-600"></div>
                        <div className="aspect-square w-8 bg-green-600"></div>
                        <div className="aspect-square w-8 bg-green-600"></div>

                        {/* Row C (2) - C4, C11 */}
                        <div className="aspect-square w-8 bg-green-600"></div>
                        <div className="aspect-square w-8 bg-green-600"></div>
                        <div className="aspect-square w-8 bg-green-600"></div>
                        <div className="flex aspect-square w-8 items-center justify-center bg-green-600">
                          <div className="h-4 w-4 rounded-full border-2 border-gray-400 bg-white shadow-lg"></div>
                        </div>
                        <div className="aspect-square w-8 border-r-4 border-r-white bg-green-600"></div>
                        <div className="aspect-square w-8 bg-green-600"></div>
                        <div className="aspect-square w-8 bg-green-600"></div>
                        <div className="aspect-square w-8 bg-green-600"></div>
                        <div className="aspect-square w-8 border-r-4 border-r-white bg-green-600"></div>
                        <div className="aspect-square w-8 bg-green-600"></div>
                        <div className="flex aspect-square w-8 items-center justify-center bg-green-600">
                          <div className="h-4 w-4 rounded-full border-2 border-gray-600 bg-gray-800 shadow-lg"></div>
                        </div>
                        <div className="aspect-square w-8 bg-green-600"></div>
                        <div className="aspect-square w-8 bg-green-600"></div>
                        <div className="aspect-square w-8 bg-green-600"></div>

                        {/* Row D (3) - Goal area starts - D5, D10 */}
                        <div className="aspect-square w-8 border-t-4 border-r-4 border-t-white border-r-white bg-green-600"></div>
                        <div className="aspect-square w-8 bg-green-600"></div>
                        <div className="aspect-square w-8 bg-green-600"></div>
                        <div className="aspect-square w-8 bg-green-600"></div>
                        <div className="flex aspect-square w-8 items-center justify-center border-r-4 border-r-white bg-green-600">
                          <div className="h-4 w-4 rounded-full border-2 border-gray-400 bg-white shadow-lg"></div>
                        </div>
                        <div className="aspect-square w-8 bg-green-600"></div>
                        <div className="aspect-square w-8 bg-green-600"></div>
                        <div className="aspect-square w-8 bg-green-600"></div>
                        <div className="aspect-square w-8 border-r-4 border-r-white bg-green-600"></div>
                        <div className="flex aspect-square w-8 items-center justify-center bg-green-600">
                          <div className="h-4 w-4 rounded-full border-2 border-gray-600 bg-gray-800 shadow-lg"></div>
                        </div>
                        <div className="aspect-square w-8 bg-green-600"></div>
                        <div className="aspect-square w-8 bg-green-600"></div>
                        <div className="aspect-square w-8 bg-green-600"></div>
                        <div className="aspect-square w-8 border-t-4 border-l-4 border-t-white border-l-white bg-green-600"></div>

                        {/* Row E (4) - Goal area continues - E3, E7, E8, E12 */}
                        <div className="flex aspect-square w-8 items-center justify-center border-r-4 border-r-white bg-green-600">
                          <div className="h-4 w-4 rounded-full border-4 border-yellow-400 bg-white shadow-lg"></div>
                        </div>
                        <div className="aspect-square w-8 bg-green-600"></div>
                        <div className="flex aspect-square w-8 items-center justify-center bg-green-600">
                          <div className="h-4 w-4 rounded-full border-2 border-gray-400 bg-white shadow-lg"></div>
                        </div>
                        <div className="aspect-square w-8 bg-green-600"></div>
                        <div className="aspect-square w-8 border-r-4 border-r-white bg-green-600"></div>
                        <div className="aspect-square w-8 bg-green-600"></div>
                        <div className="flex aspect-square w-8 items-center justify-center bg-green-600">
                          <div className="h-4 w-4 rounded-full border-2 border-gray-400 bg-white shadow-lg"></div>
                        </div>
                        <div className="flex aspect-square w-8 items-center justify-center bg-green-600">
                          <div className="h-4 w-4 rounded-full border-2 border-gray-600 bg-gray-800 shadow-lg"></div>
                        </div>
                        <div className="aspect-square w-8 border-r-4 border-r-white bg-green-600"></div>
                        <div className="aspect-square w-8 bg-green-600"></div>
                        <div className="aspect-square w-8 bg-green-600"></div>
                        <div className="flex aspect-square w-8 items-center justify-center bg-green-600">
                          <div className="h-4 w-4 rounded-full border-2 border-gray-600 bg-gray-800 shadow-lg"></div>
                        </div>
                        <div className="aspect-square w-8 bg-green-600"></div>
                        <div className="flex aspect-square w-8 items-center justify-center border-l-4 border-l-white bg-green-600">
                          <div className="h-4 w-4 rounded-full border-4 border-yellow-400 bg-gray-800 shadow-lg"></div>
                        </div>

                        {/* Row F (5) - F3, F7, F8, F12 */}
                        <div className="aspect-square w-8 border-r-4 border-r-white bg-green-600"></div>
                        <div className="aspect-square w-8 bg-green-600"></div>
                        <div className="flex aspect-square w-8 items-center justify-center bg-green-600">
                          <div className="h-4 w-4 rounded-full border-2 border-gray-400 bg-white shadow-lg"></div>
                        </div>
                        <div className="aspect-square w-8 bg-green-600"></div>
                        <div className="aspect-square w-8 border-r-4 border-r-white bg-green-600"></div>
                        <div className="aspect-square w-8 bg-green-600"></div>
                        <div className="flex aspect-square w-8 items-center justify-center bg-green-600">
                          <div className="h-4 w-4 rounded-full border-2 border-gray-400 bg-white shadow-lg"></div>
                        </div>
                        <div className="flex aspect-square w-8 items-center justify-center bg-green-600">
                          <div className="h-4 w-4 rounded-full border-2 border-gray-600 bg-gray-800 shadow-lg"></div>
                        </div>
                        <div className="aspect-square w-8 border-r-4 border-r-white bg-green-600"></div>
                        <div className="aspect-square w-8 bg-green-600"></div>
                        <div className="aspect-square w-8 bg-green-600"></div>
                        <div className="flex aspect-square w-8 items-center justify-center bg-green-600">
                          <div className="h-4 w-4 rounded-full border-2 border-gray-600 bg-gray-800 shadow-lg"></div>
                        </div>
                        <div className="aspect-square w-8 bg-green-600"></div>
                        <div className="aspect-square w-8 border-l-4 border-l-white bg-green-600"></div>

                        {/* Row G (6) - Goal area ends - G5, G10 */}
                        <div className="aspect-square w-8 border-r-4 border-b-4 border-r-white border-b-white bg-green-600"></div>
                        <div className="aspect-square w-8 bg-green-600"></div>
                        <div className="aspect-square w-8 bg-green-600"></div>
                        <div className="aspect-square w-8 bg-green-600"></div>
                        <div className="flex aspect-square w-8 items-center justify-center border-r-4 border-r-white bg-green-600">
                          <div className="h-4 w-4 rounded-full border-2 border-gray-400 bg-white shadow-lg"></div>
                        </div>
                        <div className="aspect-square w-8 bg-green-600"></div>
                        <div className="aspect-square w-8 bg-green-600"></div>
                        <div className="aspect-square w-8 bg-green-600"></div>
                        <div className="aspect-square w-8 border-r-4 border-r-white bg-green-600"></div>
                        <div className="flex aspect-square w-8 items-center justify-center bg-green-600">
                          <div className="h-4 w-4 rounded-full border-2 border-gray-600 bg-gray-800 shadow-lg"></div>
                        </div>
                        <div className="aspect-square w-8 bg-green-600"></div>
                        <div className="aspect-square w-8 bg-green-600"></div>
                        <div className="aspect-square w-8 bg-green-600"></div>
                        <div className="aspect-square w-8 border-b-4 border-l-4 border-b-white border-l-white bg-green-600"></div>

                        {/* Row H (7) - H4, H11 */}
                        <div className="aspect-square w-8 bg-green-600"></div>
                        <div className="aspect-square w-8 bg-green-600"></div>
                        <div className="aspect-square w-8 bg-green-600"></div>
                        <div className="flex aspect-square w-8 items-center justify-center bg-green-600">
                          <div className="h-4 w-4 rounded-full border-2 border-gray-400 bg-white shadow-lg"></div>
                        </div>
                        <div className="aspect-square w-8 border-r-4 border-r-white bg-green-600"></div>
                        <div className="aspect-square w-8 bg-green-600"></div>
                        <div className="aspect-square w-8 bg-green-600"></div>
                        <div className="aspect-square w-8 bg-green-600"></div>
                        <div className="aspect-square w-8 border-r-4 border-r-white bg-green-600"></div>
                        <div className="aspect-square w-8 bg-green-600"></div>
                        <div className="flex aspect-square w-8 items-center justify-center bg-green-600">
                          <div className="h-4 w-4 rounded-full border-2 border-gray-600 bg-gray-800 shadow-lg"></div>
                        </div>
                        <div className="aspect-square w-8 bg-green-600"></div>
                        <div className="aspect-square w-8 bg-green-600"></div>
                        <div className="aspect-square w-8 bg-green-600"></div>

                        {/* Row I (8) - I6, I9 */}
                        <div className="aspect-square w-8 bg-green-600"></div>
                        <div className="aspect-square w-8 bg-green-600"></div>
                        <div className="aspect-square w-8 bg-green-600"></div>
                        <div className="aspect-square w-8 bg-green-600"></div>
                        <div className="aspect-square w-8 border-r-4 border-r-white bg-green-600"></div>
                        <div className="flex aspect-square w-8 items-center justify-center bg-green-600">
                          <div className="h-4 w-4 rounded-full border-2 border-gray-400 bg-white shadow-lg"></div>
                        </div>
                        <div className="aspect-square w-8 bg-green-600"></div>
                        <div className="aspect-square w-8 bg-green-600"></div>
                        <div className="flex aspect-square w-8 items-center justify-center border-r-4 border-r-white bg-green-600">
                          <div className="h-4 w-4 rounded-full border-2 border-gray-600 bg-gray-800 shadow-lg"></div>
                        </div>
                        <div className="aspect-square w-8 bg-green-600"></div>
                        <div className="aspect-square w-8 bg-green-600"></div>
                        <div className="aspect-square w-8 bg-green-600"></div>
                        <div className="aspect-square w-8 bg-green-600"></div>
                        <div className="aspect-square w-8 bg-green-600"></div>

                        {/* Row J (9) */}
                        <div className="aspect-square w-8 bg-green-600"></div>
                        <div className="aspect-square w-8 bg-green-600"></div>
                        <div className="aspect-square w-8 bg-green-600"></div>
                        <div className="aspect-square w-8 bg-green-600"></div>
                        <div className="aspect-square w-8 border-r-4 border-r-white bg-green-600"></div>
                        <div className="aspect-square w-8 bg-green-600"></div>
                        <div className="aspect-square w-8 bg-green-600"></div>
                        <div className="aspect-square w-8 bg-green-600"></div>
                        <div className="aspect-square w-8 border-r-4 border-r-white bg-green-600"></div>
                        <div className="aspect-square w-8 bg-green-600"></div>
                        <div className="aspect-square w-8 bg-green-600"></div>
                        <div className="aspect-square w-8 bg-green-600"></div>
                        <div className="aspect-square w-8 bg-green-600"></div>
                        <div className="aspect-square w-8 bg-green-600"></div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <div className="mb-6 rounded-lg bg-blue-50 p-6">
              <h3 className="mb-3 text-lg font-semibold">Initial Formation</h3>
              <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                <div>
                  <h4 className="mb-2 font-medium text-blue-800">White Team</h4>
                  <p className="mb-1 text-sm text-gray-600">
                    • Piece with Ball: F3
                  </p>
                  <p className="mb-1 text-sm text-gray-600">
                    • Other Pieces: E3, C4, H4, D5, G5, B6, I6, E7, F7
                  </p>
                  <p className="text-sm text-gray-600">
                    • Goalie: In goal circle
                  </p>
                </div>
                <div>
                  <h4 className="mb-2 font-medium text-red-800">Black Team</h4>
                  <p className="mb-1 text-sm text-gray-600">
                    • Piece with Ball: E12
                  </p>
                  <p className="mb-1 text-sm text-gray-600">
                    • Other Pieces: F12, H11, C11, G10, D10, I9, B9, F8, E8
                  </p>
                  <p className="text-sm text-gray-600">
                    • Goalie: In goal circle
                  </p>
                </div>
              </div>
            </div>
          </section>

          {/* Core Concepts */}
          <section className="mb-12">
            <h2 className="mb-4 text-2xl font-bold text-gray-800">
              3. Core Concepts
            </h2>

            <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
              <div className="rounded-lg bg-gray-50 p-4">
                <h3 className="mb-2 font-semibold">Turn</h3>
                <p className="text-sm text-gray-600">
                  A turn consists of a single player performing one valid action
                  with one of their pieces.
                </p>
              </div>
              <div className="rounded-lg bg-gray-50 p-4">
                <h3 className="mb-2 font-semibold">Possession</h3>
                <p className="text-sm text-gray-600">
                  A piece has possession of the ball if it occupies the same
                  square as the ball.
                </p>
              </div>
              <div className="rounded-lg bg-gray-50 p-4">
                <h3 className="mb-2 font-semibold">Board Halves</h3>
                <p className="text-sm text-gray-600">
                  White&apos;s Half: Columns 1-7, Black&apos;s Half: Columns
                  8-14
                </p>
              </div>
              <div className="rounded-lg bg-gray-50 p-4">
                <h3 className="mb-2 font-semibold">Facing Direction</h3>
                <div className="mt-2 flex items-center space-x-2">
                  <ArrowUp className="h-4 w-4 text-blue-600" />
                  <ArrowDown className="h-4 w-4 text-red-600" />
                  <ArrowLeft className="h-4 w-4 text-gray-600" />
                  <ArrowRight className="h-4 w-4 text-gray-600" />
                  <span className="text-xs text-gray-500">
                    Goal | Own | Left | Right
                  </span>
                </div>
              </div>
            </div>
          </section>

          {/* Player Actions */}
          <section className="mb-12">
            <h2 className="mb-4 text-2xl font-bold text-gray-800">
              4. Player Actions
            </h2>

            <div className="space-y-6">
              {/* Move Without Ball */}
              <div className="rounded-lg border border-gray-200 p-6">
                <h3 className="mb-4 text-lg font-semibold text-blue-800">
                  Action A: Move a Piece Without the Ball
                </h3>
                <div className="mb-4 rounded-lg bg-blue-50 p-4">
                  <h4 className="mb-2 font-medium">Movement Rules</h4>
                  <ul className="space-y-1 text-sm text-gray-600">
                    <li>
                      • Forward Movement: Up to 3 squares (towards
                      opponent&apos;s goal)
                    </li>
                    <li>
                      • Backward Movement: Up to 2 squares (towards own goal)
                    </li>
                    <li>• Horizontal Movement: Up to 2 squares (same row)</li>
                    <li>• Cannot move through other pieces</li>
                  </ul>
                </div>

                {/* Movement Diagram */}
                <div className="rounded-lg border bg-white p-4">
                  <h4 className="mb-3 text-center font-medium">
                    Movement Range Example
                  </h4>
                  <div className="flex justify-center">
                    <div className="grid grid-cols-5 gap-[1px] rounded bg-green-700 p-2 shadow-lg">
                      {Array.from({ length: 25 }, (_, i) => {
                        const row = Math.floor(i / 5);
                        const col = i % 5;
                        const isCenter = row === 2 && col === 2;
                        const isForward = row === 0 && col === 2;
                        const isBackward = row === 4 && col === 2;
                        const isHorizontal =
                          row === 2 && (col === 0 || col === 4);

                        return (
                          <div
                            key={i}
                            className={`flex aspect-square items-center justify-center text-xs font-bold ${
                              isCenter
                                ? "bg-blue-600 text-white shadow-lg"
                                : isForward
                                  ? "bg-green-300 text-green-800 shadow-lg"
                                  : isBackward
                                    ? "bg-yellow-300 text-yellow-800 shadow-lg"
                                    : isHorizontal
                                      ? "bg-orange-300 text-orange-800 shadow-lg"
                                      : "bg-green-600"
                            }`}
                          >
                            {isCenter
                              ? "P"
                              : isForward
                                ? "3"
                                : isBackward
                                  ? "2"
                                  : isHorizontal
                                    ? "2"
                                    : ""}
                          </div>
                        );
                      })}
                    </div>
                  </div>
                  <div className="mt-2 text-center text-xs text-gray-600">
                    <span className="mr-1 inline-block h-3 w-3 bg-green-300 shadow-sm"></span>
                    Forward (3)
                    <span className="mr-1 ml-3 inline-block h-3 w-3 bg-yellow-300 shadow-sm"></span>
                    Backward (2)
                    <span className="mr-1 ml-3 inline-block h-3 w-3 bg-orange-300 shadow-sm"></span>
                    Horizontal (2)
                  </div>
                </div>
              </div>

              {/* Use Piece With Ball */}
              <div className="rounded-lg border border-gray-200 p-6">
                <h3 className="mb-4 text-lg font-semibold text-green-800">
                  Action B: Use a Piece With the Ball
                </h3>
                <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
                  <div className="rounded-lg bg-green-50 p-4">
                    <h4 className="mb-2 font-medium">1. Dribble</h4>
                    <p className="text-sm text-gray-600">
                      Move one square in any direction, then set facing
                      direction.
                    </p>
                  </div>
                  <div className="rounded-lg bg-green-50 p-4">
                    <h4 className="mb-2 font-medium">2. Turn on Spot</h4>
                    <p className="text-sm text-gray-600">
                      Change facing direction without moving.
                    </p>
                  </div>
                  <div className="rounded-lg bg-green-50 p-4">
                    <h4 className="mb-2 font-medium">3. Pass the Ball</h4>
                    <p className="text-sm text-gray-600">
                      Send ball to friendly piece or empty square.
                    </p>
                  </div>
                </div>
              </div>

              {/* Shooting */}
              <div className="rounded-lg border border-gray-200 p-6">
                <h3 className="mb-4 text-lg font-semibold text-red-800">
                  Action C: Shoot the Ball
                </h3>
                <div className="rounded-lg bg-red-50 p-4">
                  <p className="mb-2 text-sm text-gray-600">
                    Shooting is a special type of pass that can only be
                    initiated from the opponent&apos;s Shooting Zone.
                  </p>
                  <div className="text-sm text-gray-600">
                    <strong>Shooting Zones:</strong>
                    <br />• White&apos;s Shooting Zone: Columns 10-14
                    <br />• Black&apos;s Shooting Zone: Columns 1-5
                  </div>
                </div>
              </div>

              {/* Tackling */}
              <div className="rounded-lg border border-gray-200 p-6">
                <h3 className="mb-4 text-lg font-semibold text-purple-800">
                  Action D: Tackle (Steal the Ball)
                </h3>
                <div className="mb-4 rounded-lg bg-purple-50 p-4">
                  <h4 className="mb-2 font-medium">Tackle Rules</h4>
                  <ul className="space-y-1 text-sm text-gray-600">
                    <li>• Must be adjacent to opponent piece with ball</li>
                    <li>• Can only tackle from front or side (not behind)</li>
                    <li>• Tackling piece and opponent piece swap positions</li>
                    <li>• Tackling piece gains possession</li>
                  </ul>
                </div>

                {/* Tackle Diagram */}
                <div className="rounded-lg border bg-white p-4">
                  <h4 className="mb-3 text-center font-medium">
                    Legal Tackle Positions
                  </h4>
                  <div className="flex justify-center">
                    <div className="grid grid-cols-3 gap-[1px] rounded bg-green-700 p-2 shadow-lg">
                      {Array.from({ length: 9 }, (_, i) => {
                        const row = Math.floor(i / 3);
                        const col = i % 3;
                        const isCenter = row === 1 && col === 1;
                        const isBehind = row === 2 && col === 1;
                        const isLegal = !isCenter && !isBehind;

                        return (
                          <div
                            key={i}
                            className={`flex aspect-square items-center justify-center text-lg ${
                              isCenter
                                ? "bg-red-600 text-white shadow-lg"
                                : isBehind
                                  ? "bg-red-300 text-red-800 shadow-lg"
                                  : isLegal
                                    ? "bg-green-300 text-green-800 shadow-lg"
                                    : "bg-green-600"
                            }`}
                          >
                            {isCenter
                              ? "🏃"
                              : isBehind
                                ? "❌"
                                : isLegal
                                  ? "✓"
                                  : ""}
                          </div>
                        );
                      })}
                    </div>
                  </div>
                  <div className="mt-2 text-center text-xs text-gray-600">
                    <span className="mr-1 inline-block h-3 w-3 bg-green-300 shadow-sm"></span>
                    Legal Tackle
                    <span className="mr-1 ml-3 inline-block h-3 w-3 bg-red-300 shadow-sm"></span>
                    Illegal (Behind)
                  </div>
                </div>
              </div>
            </div>
          </section>

          {/* Passing Rules */}
          <section className="mb-12">
            <h2 className="mb-4 text-2xl font-bold text-gray-800">
              5. Detailed Passing Rules
            </h2>

            <div className="space-y-6">
              <div className="rounded-lg bg-blue-50 p-6">
                <h3 className="mb-4 text-lg font-semibold">
                  Passing Zone (90-degree cone)
                </h3>
                <div className="rounded-lg border bg-white p-4">
                  <div className="mb-4 flex justify-center">
                    <div className="grid grid-cols-5 gap-[1px] rounded bg-green-700 p-2 shadow-lg">
                      {Array.from({ length: 25 }, (_, i) => {
                        const row = Math.floor(i / 5);
                        const col = i % 5;
                        const isCenter = row === 2 && col === 2;
                        const isPassZone =
                          row <= 1 && Math.abs(col - 2) <= 2 - row;

                        return (
                          <div
                            key={i}
                            className={`flex aspect-square items-center justify-center text-lg ${
                              isCenter
                                ? "bg-blue-600 text-white shadow-lg"
                                : isPassZone
                                  ? "bg-green-300 text-green-800 shadow-lg"
                                  : "bg-green-600"
                            }`}
                          >
                            {isCenter ? "⚽" : isPassZone ? "✓" : ""}
                          </div>
                        );
                      })}
                    </div>
                  </div>
                  <p className="text-center text-sm text-gray-600">
                    Piece facing opponent&apos;s goal can pass within 90-degree
                    forward cone
                  </p>
                </div>
              </div>

              <div className="rounded-lg bg-yellow-50 p-6">
                <h3 className="mb-4 text-lg font-semibold">
                  Consecutive Passes
                </h3>
                <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                  <div>
                    <h4 className="mb-2 font-medium">Rules</h4>
                    <ul className="space-y-1 text-sm text-gray-600">
                      <li>• Can make 2 passes in one turn</li>
                      <li>• First receiver cannot have moved</li>
                      <li>• Second receiver cannot move to receive</li>
                      <li>• Entire sequence counts as one turn</li>
                    </ul>
                  </div>
                  <div className="rounded-lg border bg-green-700 p-3 shadow-lg">
                    <div className="mb-2 text-center text-sm font-medium text-white">
                      Pass Sequence
                    </div>
                    <div className="flex items-center justify-center space-x-2 text-xs">
                      <div className="flex h-6 w-6 items-center justify-center rounded-full bg-blue-600 text-white shadow-lg">
                        A
                      </div>
                      <ArrowRight className="h-4 w-4 text-white" />
                      <div className="flex h-6 w-6 items-center justify-center rounded-full bg-green-600 text-white shadow-lg">
                        B
                      </div>
                      <ArrowRight className="h-4 w-4 text-white" />
                      <div className="flex h-6 w-6 items-center justify-center rounded-full bg-purple-600 text-white shadow-lg">
                        C
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              <div className="rounded-lg bg-orange-50 p-6">
                <h3 className="mb-4 text-lg font-semibold">Chip Pass</h3>
                <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                  <div>
                    <h4 className="mb-2 font-medium">Rules</h4>
                    <ul className="space-y-1 text-sm text-gray-600">
                      <li>• Can pass over any number of pieces</li>
                      <li>
                        • Blocked if opponent piece is adjacent and on pass line
                      </li>
                      <li>• Cannot make 2 consecutive chip passes</li>
                    </ul>
                  </div>
                  <div className="rounded-lg border bg-green-700 p-3 shadow-lg">
                    <div className="mb-2 text-center text-sm font-medium text-white">
                      Chip Pass Example
                    </div>
                    <div className="flex items-center justify-center space-x-1 text-xs">
                      <div className="h-4 w-4 rounded-full bg-blue-600 shadow-lg"></div>
                      <div className="text-yellow-300">~</div>
                      <div className="h-4 w-4 rounded-full bg-red-600 shadow-lg"></div>
                      <div className="text-yellow-300">~</div>
                      <div className="h-4 w-4 rounded-full bg-blue-600 shadow-lg"></div>
                    </div>
                    <div className="mt-1 text-center text-xs text-gray-300">
                      Ball flies over opponent
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </section>

          {/* Game End Conditions */}
          <section className="mb-12">
            <h2 className="mb-4 text-2xl font-bold text-gray-800">
              6. Game End Conditions
            </h2>

            <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
              <div className="rounded-lg bg-green-50 p-6">
                <h3 className="mb-3 text-lg font-semibold text-green-800">
                  Victory Conditions
                </h3>
                <ul className="space-y-2 text-sm text-gray-600">
                  <li>• First to reach goal limit</li>
                  <li>• Most goals after move limit</li>
                  <li>• Most goals when time expires</li>
                </ul>
              </div>

              <div className="rounded-lg bg-yellow-50 p-6">
                <h3 className="mb-3 text-lg font-semibold text-yellow-800">
                  Draw Conditions
                </h3>
                <ul className="space-y-2 text-sm text-gray-600">
                  <li>• 50 turns without ball movement</li>
                  <li>• Illegal move repetition (3 times)</li>
                </ul>
              </div>
            </div>
          </section>

          {/* Special Rules */}
          <section className="mb-12">
            <h2 className="mb-4 text-2xl font-bold text-gray-800">
              7. Special Rules
            </h2>

            <div className="space-y-6">
              <div className="rounded-lg border border-gray-200 p-6">
                <h3 className="mb-4 text-lg font-semibold text-orange-800">
                  Offside Rule
                </h3>
                <div className="rounded-lg bg-orange-50 p-4">
                  <p className="mb-2 text-sm text-gray-600">
                    A piece is offside if it is closer to the opponent&apos;s
                    goal line than both the ball and the second-to-last opposing
                    piece when a pass is made to it.
                  </p>
                  <p className="text-sm text-gray-600">
                    <strong>Note:</strong> The goalkeeper counts as a piece for
                    this rule.
                  </p>
                </div>
              </div>

              <div className="rounded-lg border border-gray-200 p-6">
                <h3 className="mb-4 text-lg font-semibold text-blue-800">
                  Goalkeeper Special Rules
                </h3>
                <div className="rounded-lg bg-blue-50 p-4">
                  <ul className="space-y-2 text-sm text-gray-600">
                    <li>• Only piece allowed in goal zone squares</li>
                    <li>
                      • Automatically blocks shots that pass through their
                      square
                    </li>
                    <li>• Cannot be chip passed over from the last row</li>
                    <li>• Acts like normal piece outside goal zone</li>
                  </ul>
                </div>
              </div>
            </div>
          </section>

          {/* Back to Home */}
          <div className="border-t border-gray-200 pt-8 text-center">
            <Link
              href="/"
              className="rounded-lg bg-green-600 px-8 py-3 font-semibold text-white transition-colors hover:bg-green-700"
            >
              Ready to Play Taka?
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
};

export default RulesPage;
