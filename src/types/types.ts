// Types
export type PlayerColor = "white" | "black";

export type TutorialStep =
  | "welcome"
  | "basic_movement"
  | "turning"
  | "movement_with_ball"
  | "passing"
  | "completed";

export type FacingDirection = "north" | "south" | "west" | "east";

/**
 * nothing - no pointer actions
 * piece - piece on square, clickable
 * movement - blue dot, clickable
 * turn target - turn, clickable
 */
export type SquareType = "nothing" | "piece" | "movement" | "turn_target" | "pass_target";
