import React from "react";

const StaticGameBoard: React.FC = () => {
  return (
    <div className="flex w-full items-center justify-center">
      <div className="shadow-game-board animate-slide-up grid w-full max-w-4xl origin-center -rotate-90 grid-cols-14 grid-rows-10 gap-[1px] rounded bg-green-700 p-2 opacity-0 md:max-w-4xl md:rotate-0">
        {/* Row A (0) */}
        <div className="aspect-square bg-green-600" />
        <div className="aspect-square bg-green-600" />
        <div className="aspect-square bg-green-600" />
        <div className="aspect-square bg-green-600" />
        <div className="aspect-square border-r-4 border-r-white bg-green-600" />
        <div className="aspect-square bg-green-600" />
        <div className="aspect-square bg-green-600" />
        <div className="aspect-square bg-green-600" />
        <div className="aspect-square border-r-4 border-r-white bg-green-600" />
        <div className="aspect-square bg-green-600" />
        <div className="aspect-square bg-green-600" />
        <div className="aspect-square bg-green-600" />
        <div className="aspect-square bg-green-600" />
        <div className="aspect-square bg-green-600" />

        {/* Row B (1) - B6 */}
        <div className="aspect-square bg-green-600" />
        <div className="aspect-square bg-green-600" />
        <div className="aspect-square bg-green-600" />
        <div className="aspect-square bg-green-600" />
        <div className="aspect-square border-r-4 border-r-white bg-green-600" />
        <div className="flex aspect-square items-center justify-center bg-green-600">
          <div className="h-8 w-8 rounded-full border-2 border-gray-400 bg-white shadow-lg" />
        </div>
        <div className="aspect-square bg-green-600" />
        <div className="aspect-square bg-green-600" />
        <div className="flex aspect-square items-center justify-center border-r-4 border-r-white bg-green-600">
          <div className="h-8 w-8 rounded-full border-2 border-gray-600 bg-gray-800 shadow-lg" />
        </div>
        <div className="aspect-square bg-green-600" />
        <div className="aspect-square bg-green-600" />
        <div className="aspect-square bg-green-600" />
        <div className="aspect-square bg-green-600" />
        <div className="aspect-square bg-green-600" />

        {/* Row C (2) - C4, C11 */}
        <div className="aspect-square bg-green-600" />
        <div className="aspect-square bg-green-600" />
        <div className="aspect-square bg-green-600" />
        <div className="flex aspect-square items-center justify-center bg-green-600">
          <div className="h-8 w-8 rounded-full border-2 border-gray-400 bg-white shadow-lg" />
        </div>
        <div className="aspect-square border-r-4 border-r-white bg-green-600" />
        <div className="aspect-square bg-green-600" />
        <div className="aspect-square bg-green-600" />
        <div className="aspect-square bg-green-600" />
        <div className="aspect-square border-r-4 border-r-white bg-green-600" />
        <div className="aspect-square bg-green-600" />
        <div className="flex aspect-square items-center justify-center bg-green-600">
          <div className="h-8 w-8 rounded-full border-2 border-gray-600 bg-gray-800 shadow-lg" />
        </div>
        <div className="aspect-square bg-green-600" />
        <div className="aspect-square bg-green-600" />
        <div className="aspect-square bg-green-600" />

        {/* Row D (3) - Goal area starts - D5, D10 */}
        <div className="aspect-square border-t-4 border-r-4 border-t-white border-r-white bg-green-600" />
        <div className="aspect-square bg-green-600" />
        <div className="aspect-square bg-green-600" />
        <div className="aspect-square bg-green-600" />
        <div className="flex aspect-square items-center justify-center border-r-4 border-r-white bg-green-600">
          <div className="h-8 w-8 rounded-full border-2 border-gray-400 bg-white shadow-lg" />
        </div>
        <div className="aspect-square bg-green-600" />
        <div className="aspect-square bg-green-600" />
        <div className="aspect-square bg-green-600" />
        <div className="aspect-square border-r-4 border-r-white bg-green-600" />
        <div className="flex aspect-square items-center justify-center bg-green-600">
          <div className="h-8 w-8 rounded-full border-2 border-gray-600 bg-gray-800 shadow-lg" />
        </div>
        <div className="aspect-square bg-green-600" />
        <div className="aspect-square bg-green-600" />
        <div className="aspect-square bg-green-600" />
        <div className="aspect-square border-t-4 border-l-4 border-t-white border-l-white bg-green-600" />

        {/* Row E (4) - Goal area continues - E3, E7, E8, E12 */}
        <div className="flex aspect-square items-center justify-center border-r-4 border-r-white bg-green-600">
          <div className="h-8 w-8 rounded-full border-4 border-yellow-400 bg-white shadow-lg" />
        </div>
        <div className="aspect-square bg-green-600" />
        <div className="flex aspect-square items-center justify-center bg-green-600">
          <div className="h-8 w-8 rounded-full border-2 border-gray-400 bg-white shadow-lg" />
        </div>
        <div className="aspect-square bg-green-600" />
        <div className="aspect-square border-r-4 border-r-white bg-green-600" />
        <div className="aspect-square bg-green-600" />
        <div className="flex aspect-square items-center justify-center bg-green-600">
          <div className="h-8 w-8 rounded-full border-2 border-gray-400 bg-white shadow-lg" />
        </div>
        <div className="flex aspect-square items-center justify-center bg-green-600">
          <div className="h-8 w-8 rounded-full border-2 border-gray-600 bg-gray-800 shadow-lg" />
        </div>
        <div className="aspect-square border-r-4 border-r-white bg-green-600" />
        <div className="aspect-square bg-green-600" />
        <div className="aspect-square bg-green-600" />
        <div className="flex aspect-square items-center justify-center bg-green-600">
          <div className="h-8 w-8 rounded-full border-2 border-gray-600 bg-gray-800 shadow-lg" />
        </div>
        <div className="aspect-square bg-green-600" />
        <div className="flex aspect-square items-center justify-center border-l-4 border-l-white bg-green-600">
          <div className="h-8 w-8 rounded-full border-4 border-yellow-400 bg-gray-800 shadow-lg" />
        </div>

        {/* Row F (5) - F3, F7, F8, F12 */}
        <div className="aspect-square border-r-4 border-r-white bg-green-600" />
        <div className="aspect-square bg-green-600" />
        <div className="flex aspect-square items-center justify-center bg-green-600">
          <div className="h-8 w-8 rounded-full border-2 border-gray-400 bg-white shadow-lg" />
        </div>
        <div className="aspect-square bg-green-600" />
        <div className="aspect-square border-r-4 border-r-white bg-green-600" />
        <div className="aspect-square bg-green-600" />
        <div className="flex aspect-square items-center justify-center bg-green-600">
          <div className="h-8 w-8 rounded-full border-2 border-gray-400 bg-white shadow-lg" />
        </div>
        <div className="flex aspect-square items-center justify-center bg-green-600">
          <div className="h-8 w-8 rounded-full border-2 border-gray-600 bg-gray-800 shadow-lg" />
        </div>
        <div className="aspect-square border-r-4 border-r-white bg-green-600" />
        <div className="aspect-square bg-green-600" />
        <div className="aspect-square bg-green-600" />
        <div className="flex aspect-square items-center justify-center bg-green-600">
          <div className="h-8 w-8 rounded-full border-2 border-gray-600 bg-gray-800 shadow-lg" />
        </div>
        <div className="aspect-square bg-green-600" />
        <div className="aspect-square border-l-4 border-l-white bg-green-600" />

        {/* Row G (6) - Goal area ends - G5, G10 */}
        <div className="aspect-square border-r-4 border-b-4 border-r-white border-b-white bg-green-600" />
        <div className="aspect-square bg-green-600" />
        <div className="aspect-square bg-green-600" />
        <div className="aspect-square bg-green-600" />
        <div className="flex aspect-square items-center justify-center border-r-4 border-r-white bg-green-600">
          <div className="h-8 w-8 rounded-full border-2 border-gray-400 bg-white shadow-lg" />
        </div>
        <div className="aspect-square bg-green-600" />
        <div className="aspect-square bg-green-600" />
        <div className="aspect-square bg-green-600" />
        <div className="aspect-square border-r-4 border-r-white bg-green-600" />
        <div className="flex aspect-square items-center justify-center bg-green-600">
          <div className="h-8 w-8 rounded-full border-2 border-gray-600 bg-gray-800 shadow-lg" />
        </div>
        <div className="aspect-square bg-green-600" />
        <div className="aspect-square bg-green-600" />
        <div className="aspect-square bg-green-600" />
        <div className="aspect-square border-b-4 border-l-4 border-b-white border-l-white bg-green-600" />

        {/* Row H (7) - H4, H11 */}
        <div className="aspect-square bg-green-600" />
        <div className="aspect-square bg-green-600" />
        <div className="aspect-square bg-green-600" />
        <div className="flex aspect-square items-center justify-center bg-green-600">
          <div className="h-8 w-8 rounded-full border-2 border-gray-400 bg-white shadow-lg" />
        </div>
        <div className="aspect-square border-r-4 border-r-white bg-green-600" />
        <div className="aspect-square bg-green-600" />
        <div className="aspect-square bg-green-600" />
        <div className="aspect-square bg-green-600" />
        <div className="aspect-square border-r-4 border-r-white bg-green-600" />
        <div className="aspect-square bg-green-600" />
        <div className="flex aspect-square items-center justify-center bg-green-600">
          <div className="h-8 w-8 rounded-full border-2 border-gray-600 bg-gray-800 shadow-lg" />
        </div>
        <div className="aspect-square bg-green-600" />
        <div className="aspect-square bg-green-600" />
        <div className="aspect-square bg-green-600" />

        {/* Row I (8) - I6, I9 */}
        <div className="aspect-square bg-green-600" />
        <div className="aspect-square bg-green-600" />
        <div className="aspect-square bg-green-600" />
        <div className="aspect-square bg-green-600" />
        <div className="aspect-square border-r-4 border-r-white bg-green-600" />
        <div className="flex aspect-square items-center justify-center bg-green-600">
          <div className="h-8 w-8 rounded-full border-2 border-gray-400 bg-white shadow-lg" />
        </div>
        <div className="aspect-square bg-green-600" />
        <div className="aspect-square bg-green-600" />
        <div className="flex aspect-square items-center justify-center border-r-4 border-r-white bg-green-600">
          <div className="h-8 w-8 rounded-full border-2 border-gray-600 bg-gray-800 shadow-lg" />
        </div>
        <div className="aspect-square bg-green-600" />
        <div className="aspect-square bg-green-600" />
        <div className="aspect-square bg-green-600" />
        <div className="aspect-square bg-green-600" />
        <div className="aspect-square bg-green-600" />

        {/* Row J (9) */}
        <div className="aspect-square bg-green-600" />
        <div className="aspect-square bg-green-600" />
        <div className="aspect-square bg-green-600" />
        <div className="aspect-square bg-green-600" />
        <div className="aspect-square border-r-4 border-r-white bg-green-600" />
        <div className="aspect-square bg-green-600" />
        <div className="aspect-square bg-green-600" />
        <div className="aspect-square bg-green-600" />
        <div className="aspect-square border-r-4 border-r-white bg-green-600" />
        <div className="aspect-square bg-green-600" />
        <div className="aspect-square bg-green-600" />
        <div className="aspect-square bg-green-600" />
        <div className="aspect-square bg-green-600" />
        <div className="aspect-square bg-green-600" />
      </div>
    </div>
  );
};

export default StaticGameBoard;
