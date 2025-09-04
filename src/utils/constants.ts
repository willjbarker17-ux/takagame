import type { PlayerColor } from "@/types/types";
import { Position } from "@/classes/Position";

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

export const WHITE_GOALIE_ACTIVATION_TARGETS: Position[] = [
  new Position(0, 3),
  new Position(0, 4),
  new Position(0, 5),
  new Position(0, 6),
  new Position(1, 3),
  new Position(1, 4),
  new Position(1, 5),
  new Position(1, 6),
  new Position(2, 3),
  new Position(2, 4),
  new Position(2, 5),
  new Position(2, 6),
  new Position(3, 2),
  new Position(3, 4),
  new Position(3, 5),
  new Position(3, 7),
];

export const BLACK_GOALIE_ACTIVATION_TARGETS: Position[] = [
  new Position(13, 3),
  new Position(13, 4),
  new Position(13, 5),
  new Position(13, 6),
  new Position(12, 3),
  new Position(12, 4),
  new Position(12, 5),
  new Position(12, 6),
  new Position(11, 3),
  new Position(11, 4),
  new Position(11, 5),
  new Position(11, 6),
  new Position(10, 2),
  new Position(10, 4),
  new Position(10, 5),
  new Position(10, 7),
];
