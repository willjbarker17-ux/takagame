// Types
import { Piece } from "@/classes/Piece";

export type PlayerColor = "white" | "black";

export type TutorialStep =
  | "welcome"
  | "basic_movement"
  | "turning"
  | "movement_with_ball"
  | "passing"
  | "consecutive_pass"
  | "ball_empty_square"
  | "completed";

export type FacingDirection = "north" | "south" | "west" | "east";

/**
 * nothing - no pointer actions
 * piece - piece on square, clickable
 * movement - blue dot, clickable
 * turn target - turn, clickable
 */
export type SquareType =
  | "nothing"
  | "piece"
  | "movement"
  | "turn_target"
  | "pass_target";

export type BoardSquareType = Piece | "ball" | null;
