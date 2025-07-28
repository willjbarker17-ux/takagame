export class Position {
  private readonly row: number; // 0-13 (1-14)
  private readonly col: number; // 0-9 (A-J)

  constructor(row: number, col: number) {
    if (row > 13 || row < 0 || col > 9 || col < 0) {
      throw new Error("Out of bounds position");
    }

    this.row = row;
    this.col = col;
  }

  getPositionCoordinates() {
    return [this.row, this.col];
  }

  isPositionInGoal() {
    return (
      this.row >= 3 && this.row <= 6 && (this.col === 0 || this.col === 13)
    );
  }

  equals(o: Position) {
    const myPos = this.getPositionCoordinates();
    const theirPos = o.getPositionCoordinates();

    return myPos[0] === theirPos[0] && myPos[1] === theirPos[1];
  }
}
