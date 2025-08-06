import type { FacingDirection, PlayerColor } from "@/types/types";
import { Position } from "@/classes/Position";
import {
  DIRECTION_VECTORS,
  FORWARD_MOVE_DISTANCE,
  OTHER_MOVE_DISTANCE,
} from "@/utils/constants";

interface PieceParams {
  id: string;
  color: PlayerColor;
  position: Position;
  hasBall: boolean;
  facingDirection?: FacingDirection;
  isGoalie?: boolean;
}

export class Piece {
  private readonly id: string;
  private readonly color: PlayerColor;
  private position: Position;
  private hasBall: boolean;
  private facingDirection: FacingDirection;
  private readonly isGoalie: boolean;

  constructor(params: PieceParams) {
    this.id = params.id;
    this.color = params.color;
    this.position = params.position;
    this.hasBall = params.hasBall;
    this.facingDirection =
      params.facingDirection ?? (params.color === "white" ? "south" : "north");
    this.isGoalie = params.isGoalie ?? false;
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
        (this.color === "white" && dRow > 0) ||
        (this.color === "black" && dRow < 0);

      const isHorizontal = dCol === 0 && dRow !== 0; // horizontal moves (same row)
      const isVertical = dRow === 0 && dCol !== 0; // vertical moves (same column)

      let maxDistance = 0;
      if (isTowardOpponentGoal) {
        maxDistance = FORWARD_MOVE_DISTANCE;
      } else if (isHorizontal || isVertical) {
        maxDistance = OTHER_MOVE_DISTANCE;
      }

      for (let distance = 1; distance <= maxDistance; distance++) {
        const newRow = curRow + dRow * distance;
        const newCol = curCol + dCol * distance;

        if (newRow < 0 || newCol < 0 || newRow > 13 || newCol > 9) break;

        const newPosition = new Position(newRow, newCol);

        // Only goalies can enter goal areas, normal pieces cannot
        if (this.isGoalie || !newPosition.isPositionInGoal()) {
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
      const newRow = curRow + dRow;
      const newCol = curCol + dCol;

      if (newRow < 0 || newCol < 0 || newRow > 13 || newCol > 9) break;

      const newPosition = new Position(newRow, newCol);

      // Only goalies can enter goal areas, normal pieces cannot
      if (this.isGoalie || !newPosition.isPositionInGoal()) {
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

  setHasBall(hasBall: boolean) {
    this.hasBall = hasBall;
  }

  getHasBall() {
    return this.hasBall;
  }

  setFacingDirection(facingDirection: FacingDirection) {
    this.facingDirection = facingDirection;
  }

  getFacingDirection() {
    return this.facingDirection;
  }

  getIsGoalie() {
    return this.isGoalie;
  }
}
