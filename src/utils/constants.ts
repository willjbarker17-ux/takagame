import type { PlayerColor } from "@/types/types";

export const BOARD_ROWS = 14;
export const BOARD_COLS = 10;

export const FORWARD_MOVE_DISTANCE = 3;
export const OTHER_MOVE_DISTANCE = 2;

export const TUTORIAL_PLAYER_COLOR: PlayerColor = "white";
export const TUTORIAL_OPPONENT_COLOR: PlayerColor = "black";

export const DIRECTION_VECTORS: [number, number][] = [
  [-1, -1], // Up-left
  [-1, 0], // Up
  [-1, 1], // Up-right
  [0, -1], // Left
  [0, 1], // Right
  [1, -1], // Down-left
  [1, 0], // Down
  [1, 1], // Down-right
];
