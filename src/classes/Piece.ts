import type { FacingDirection, PlayerColor } from "@/types/types";
import { Position } from "@/classes/Position";
import {
  DIRECTION_VECTORS,
  FORWARD_MOVE_DISTANCE,
  OTHER_MOVE_DISTANCE,
} from "@/utils/constants";

export class Piece {
  private readonly id: string;
  private readonly color: PlayerColor;
  private position: Position;
  private hasBall: boolean;
  private facingDirection: FacingDirection;
  private readonly goalie: boolean = false;

  constructor(
    id: string,
    color: PlayerColor,
    position: Position,
    hasBall: boolean,
  ) {
    this.id = id;
    this.color = color;
    this.position = position;
    this.hasBall = hasBall;
    this.facingDirection =
      color === "white" ? "towards_black_goal" : "towards_white_goal";
  }

  /**
   * Get the standard move targets for when a piece does not have the ball
   * @private
   */
  private getStandardMovementTargets(): Position[] {
    const validMoves: Position[] = [];

    const [curRow, curCol] = this.position.getPositionCoordinates();

    for (const [dRow, dCol] of DIRECTION_VECTORS) {
      const isTowardOpponentGoal =
        (this.color === "white" && dCol > 0) ||
        (this.color === "black" && dCol < 0);

      const isTowardOwnGoal =
        (this.color === "white" && dCol < 0) ||
        (this.color === "black" && dCol > 0);

      const isHorizontal = dRow === 0 && dCol !== 0; // horizontal moves (same row)
      const isVertical = dCol === 0 && dRow !== 0; // vertical moves (same column)

      let maxDistance = 0;
      if (isTowardOpponentGoal) {
        maxDistance = FORWARD_MOVE_DISTANCE;
      } else if (isTowardOwnGoal || isHorizontal || isVertical) {
        maxDistance = OTHER_MOVE_DISTANCE;
      }

      for (let distance = 1; distance <= maxDistance; distance++) {
        const newPosition = new Position(
          curRow + dRow * distance,
          curCol + dCol * distance,
        );

        if (this.goalie || (!this.goalie && !newPosition.isPositionInGoal())) {
          validMoves.push(newPosition);
        } else {
          break; // Path is blocked
        }
      }
    }

    return validMoves;
  }

  /**
   * Get movement targets for when a piece has the ball
   * @private
   */
  private getBallMovementTargets(): Position[] {
    const validMoves: Position[] = [];

    const [curRow, curCol] = this.position.getPositionCoordinates();

    for (const [dRow, dCol] of DIRECTION_VECTORS) {
      const newPosition = new Position(curRow + dRow, curCol + dCol);

      if (this.goalie || (!this.goalie && !newPosition.isPositionInGoal())) {
        validMoves.push(newPosition);
      }
    }

    return validMoves;
  }

  /**
   * Get valid movement targets. This does not account for other players positions
   */
  getMovementTargets(): Position[] {
    if (this.hasBall) return this.getBallMovementTargets();

    return this.getStandardMovementTargets();
  }

  getColor() {
    return this.color;
  }

  getPosition() {
    return this.position;
  }

  setPosition(position: Position) {
    this.position = position;
  }

  getHasBall() {
    return this.hasBall;
  }

  getFacingDirection() {
    return this.facingDirection;
  }
}
