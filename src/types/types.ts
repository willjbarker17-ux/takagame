// Types
import { Piece } from "@/classes/Piece";
import { Position } from "@/classes/Position";

export type PlayerColor = "white" | "black";

export type TutorialStep =
  | "welcome"
  | "basic_movement"
  | "turning"
  | "movement_with_ball"
  | "passing"
  | "consecutive_pass"
  | "ball_empty_square"
  | "ball_pickup"
  | "receiving_passes"
  | "chip_pass"
  | "shooting"
  | "tackling"
  | "activating_goalies"
  | "completed";

export type FacingDirection = "north" | "south" | "west" | "east";

/**
 * nothing - no pointer actions
 * piece - piece on square, clickable
 * movement - blue dot, clickable
 * turn target - turn, clickable
 * pass_target - pass to piece, clickable
 * empty_pass_target - pass to empty square, clickable
 */
export type SquareType =
  | "nothing"
  | "piece"
  | "movement"
  | "turn_target"
  | "pass_target"
  | "empty_pass_target"
  | "tackle_target";

export type BoardSquareType = Piece | "ball" | null;

export type BoardType = BoardSquareType[][];

export type PiecePositionType =
  | Position
  | "white_unactivated"
  | "black_unactivated";
