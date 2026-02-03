"""
Decision Engine Demonstration

This script demonstrates the core capabilities of the Football Decision Engine:
1. Elimination calculation
2. State scoring
3. Defensive block modeling
4. Visualization

Run with: python examples/decision_engine_demo.py
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from decision_engine import (
    # Core types
    Position,
    Player,
    # Elimination
    EliminationCalculator,
    EliminationState,
    DefenderStatus,
    # Physics
    DefensiveForceModel,
    # Scoring
    GameStateEvaluator,
    GameState,
    StateScore,
    # Blocks
    DefensiveBlock,
    BlockType,
    # Visualization
    DecisionEngineVisualizer,
    # Constants
    PITCH_LENGTH,
    PITCH_WIDTH,
    HALF_LENGTH,
)


def create_sample_scenario():
    """Create a sample attacking scenario."""
    # Ball carrier in right channel, attacking third
    ball_carrier = Player(
        id="ATK1",
        position=Position(30, 10),
        team="attack",
        max_speed=8.0,
    )

    # Supporting attackers
    attackers = [
        ball_carrier,
        Player(id="ATK2", position=Position(25, -5), team="attack"),
        Player(id="ATK3", position=Position(20, 15), team="attack"),
        Player(id="ATK4", position=Position(15, 0), team="attack"),
    ]

    # Defending team in mid-block
    defenders = [
        # Back 4
        Player(id="DEF1", position=Position(10, -15), team="defense"),  # LB
        Player(id="DEF2", position=Position(8, -5), team="defense"),    # LCB
        Player(id="DEF3", position=Position(8, 5), team="defense"),     # RCB
        Player(id="DEF4", position=Position(10, 15), team="defense"),   # RB
        # Midfield 4
        Player(id="DEF5", position=Position(20, -12), team="defense"),  # LM
        Player(id="DEF6", position=Position(18, -3), team="defense"),   # LCM
        Player(id="DEF7", position=Position(18, 3), team="defense"),    # RCM
        Player(id="DEF8", position=Position(22, 12), team="defense"),   # RM
        # Forwards 2
        Player(id="DEF9", position=Position(32, -5), team="defense"),   # LF
        Player(id="DEF10", position=Position(35, 8), team="defense"),   # RF - pressing
    ]

    return ball_carrier, attackers, defenders


def demo_elimination():
    """Demonstrate elimination calculation."""
    print("\n" + "="*60)
    print("ELIMINATION ANALYSIS")
    print("="*60)

    ball_carrier, attackers, defenders = create_sample_scenario()

    calculator = EliminationCalculator()
    state = calculator.calculate(
        ball_position=ball_carrier.position,
        ball_carrier=ball_carrier,
        defenders=defenders,
    )

    print(f"\nBall Position: ({ball_carrier.position.x:.1f}, {ball_carrier.position.y:.1f})")
    print(f"\nDefenders: {len(defenders)}")
    print(f"Eliminated: {state.eliminated_count}")
    print(f"Active: {state.active_count}")
    print(f"Elimination Ratio: {state.elimination_ratio:.1%}")

    print("\nDetailed Status:")
    print("-" * 50)
    for result in state.defenders:
        status_symbol = "X" if result.is_eliminated else "O"
        print(f"  [{status_symbol}] {result.defender.id}: {result.status.value}")
        print(f"      Position: ({result.defender.position.x:.1f}, {result.defender.position.y:.1f})")
        print(f"      Time to intervene: {result.time_to_intervene:.2f}s")
        print(f"      Time margin: {result.margin:.2f}s")

    return state


def demo_defensive_physics():
    """Demonstrate defensive force model."""
    print("\n" + "="*60)
    print("DEFENSIVE PHYSICS MODEL")
    print("="*60)

    ball_carrier, attackers, defenders = create_sample_scenario()

    model = DefensiveForceModel(
        ball_weight=1.0,
        goal_weight=0.6,
        opponent_weight=0.8,
    )

    # Pick one defender to analyze
    defender = defenders[2]  # RCB
    teammates = [d for d in defenders if d.id != defender.id]

    print(f"\nAnalyzing defender: {defender.id}")
    print(f"Current position: ({defender.position.x:.1f}, {defender.position.y:.1f})")

    forces = model.calculate_forces(
        defender=defender,
        ball_position=ball_carrier.position,
        teammates=teammates,
        opponents=attackers,
    )

    print(f"\nForces acting on {defender.id}:")
    print("-" * 50)
    for force in forces:
        print(f"  {force.force_type.value}: magnitude={force.magnitude:.2f}")

    ideal_position = model.calculate_equilibrium_position(defender, forces)
    print(f"\nIdeal position: ({ideal_position.x:.1f}, {ideal_position.y:.1f})")
    print(f"Current position: ({defender.position.x:.1f}, {defender.position.y:.1f})")
    print(f"Adjustment needed: ({ideal_position.x - defender.position.x:.1f}, "
          f"{ideal_position.y - defender.position.y:.1f})")

    # Calculate team shape
    shape = model.calculate_team_shape(
        defenders=defenders,
        ball_position=ball_carrier.position,
        opponents=attackers,
    )

    print(f"\nTeam Shape Metrics:")
    print(f"  Compactness: {shape.compactness:.1f}m")
    print(f"  Depth: {shape.depth:.1f}m")
    print(f"  Width: {shape.width:.1f}m")
    print(f"  Line heights: {[f'{h:.1f}m' for h in shape.line_heights]}")

    return shape


def demo_state_scoring():
    """Demonstrate game state evaluation."""
    print("\n" + "="*60)
    print("GAME STATE SCORING")
    print("="*60)

    ball_carrier, attackers, defenders = create_sample_scenario()

    evaluator = GameStateEvaluator()

    state = GameState(
        ball_position=ball_carrier.position,
        ball_carrier=ball_carrier,
        attackers=attackers,
        defenders=defenders,
    )

    evaluated = evaluator.evaluate(state)
    score = evaluated.score

    print(f"\nGame State Score: {score.total:.3f}")
    print("\nComponent Breakdown:")
    print("-" * 50)
    print(f"  Elimination:  {score.elimination_score:.3f} (weight: 25%)")
    print(f"  Proximity:    {score.proximity_score:.3f} (weight: 20%)")
    print(f"  Angle:        {score.angle_score:.3f} (weight: 15%)")
    print(f"  Density:      {score.density_score:.3f} (weight: 15%)")
    print(f"  Compactness:  {score.compactness_score:.3f} (weight: 10%)")
    print(f"  Action:       {score.action_score:.3f} (weight: 15%)")

    print("\nAvailable Actions (top 5):")
    print("-" * 50)
    for i, action in enumerate(evaluated.available_actions[:5]):
        print(f"  {i+1}. {action.action_type.value}")
        print(f"     Target: ({action.target.x:.1f}, {action.target.y:.1f})")
        print(f"     Success Prob: {action.success_probability:.1%}")
        print(f"     Expected Value: {action.expected_value:.3f}")

    return evaluated


def demo_defensive_blocks():
    """Demonstrate defensive block configurations."""
    print("\n" + "="*60)
    print("DEFENSIVE BLOCK MODELS")
    print("="*60)

    ball_position = Position(15, 5)

    for block_type in [BlockType.LOW, BlockType.MID, BlockType.HIGH]:
        block = DefensiveBlock(block_type)

        print(f"\n{block_type.value.upper()} BLOCK:")
        print("-" * 40)
        print(f"  Defensive line height: {block.config.defensive_line_height:.1f}m")
        print(f"  Midfield line height:  {block.config.midfield_line_height:.1f}m")
        print(f"  Forward line height:   {block.config.forward_line_height:.1f}m")
        print(f"  Press trigger:         {block.config.press_trigger_distance:.1f}m")
        print(f"  Vertical compactness:  {block.config.vertical_compactness:.1f}m")

        positions = block.calculate_positions(ball_position, "4-4-2")
        print(f"\n  Sample positions for 4-4-2:")
        for pos_name, pos in list(positions.items())[:4]:
            print(f"    {pos_name}: ({pos.x:.1f}, {pos.y:.1f})")


def demo_visualization():
    """Demonstrate visualization capabilities."""
    print("\n" + "="*60)
    print("VISUALIZATION")
    print("="*60)

    try:
        ball_carrier, attackers, defenders = create_sample_scenario()

        evaluator = GameStateEvaluator()
        state = GameState(
            ball_position=ball_carrier.position,
            ball_carrier=ball_carrier,
            attackers=attackers,
            defenders=defenders,
        )
        evaluated = evaluator.evaluate(state)

        viz = DecisionEngineVisualizer()

        # Create output directory
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'output', 'decision_engine')
        os.makedirs(output_dir, exist_ok=True)

        # 1. Game state visualization
        print("\nGenerating game state visualization...")
        fig = viz.plot_game_state(evaluated, title="Sample Attack Scenario")
        output_path = os.path.join(output_dir, "game_state.png")
        viz.save_figure(fig, output_path)
        print(f"  Saved: {output_path}")

        # 2. Elimination analysis
        print("\nGenerating elimination analysis...")
        fig = viz.plot_elimination_state(
            evaluated.elimination_state,
            attackers=attackers,
            title="Elimination Analysis",
        )
        output_path = os.path.join(output_dir, "elimination.png")
        viz.save_figure(fig, output_path)
        print(f"  Saved: {output_path}")

        # 3. Defensive block visualization
        print("\nGenerating defensive block visualization...")
        block = DefensiveBlock(BlockType.MID)
        fig = viz.plot_defensive_block(
            block,
            ball_position=ball_carrier.position,
            formation="4-4-2",
            title="Mid Block Configuration",
        )
        output_path = os.path.join(output_dir, "block.png")
        viz.save_figure(fig, output_path)
        print(f"  Saved: {output_path}")

        # 4. Action options
        print("\nGenerating action options visualization...")
        fig = viz.plot_action_options(evaluated, title="Available Actions")
        output_path = os.path.join(output_dir, "actions.png")
        viz.save_figure(fig, output_path)
        print(f"  Saved: {output_path}")

        # 5. Value heatmap
        print("\nGenerating value heatmap...")
        heatmap = evaluator.generate_value_heatmap(evaluated, grid_resolution=5.0)
        fig = viz.plot_value_heatmap(heatmap, title="Position Value Heatmap")
        output_path = os.path.join(output_dir, "heatmap.png")
        viz.save_figure(fig, output_path)
        print(f"  Saved: {output_path}")

        print(f"\nAll visualizations saved to: {output_dir}")

    except ImportError as e:
        print(f"\nVisualization skipped (missing dependency): {e}")
        print("Install with: pip install matplotlib mplsoccer")


def demo_position_comparison():
    """Demonstrate comparing different ball positions."""
    print("\n" + "="*60)
    print("POSITION COMPARISON")
    print("="*60)

    ball_carrier, attackers, defenders = create_sample_scenario()

    evaluator = GameStateEvaluator()
    base_state = GameState(
        ball_position=ball_carrier.position,
        ball_carrier=ball_carrier,
        attackers=attackers,
        defenders=defenders,
    )

    # Compare different positions
    test_positions = [
        Position(25, 0),   # Central
        Position(30, 15),  # Wide right
        Position(35, 0),   # More advanced central
        Position(40, -10), # Left channel, close to goal
        Position(20, 0),   # Deeper central
    ]

    print("\nPosition Value Comparison:")
    print("-" * 60)
    print(f"{'Position':<20} {'Score':<10} {'Elim':<10} {'Proximity':<10}")
    print("-" * 60)

    results = evaluator.compare_positions(test_positions, base_state)

    for pos, score in results:
        # Get detailed breakdown
        test_state = GameState(
            ball_position=pos,
            ball_carrier=ball_carrier,
            attackers=attackers,
            defenders=defenders,
        )
        evaluated = evaluator.evaluate(test_state)

        print(f"({pos.x:5.1f}, {pos.y:5.1f})   "
              f"{score:.3f}     "
              f"{evaluated.score.elimination_score:.3f}     "
              f"{evaluated.score.proximity_score:.3f}")


def main():
    """Run all demonstrations."""
    print("\n" + "#"*60)
    print("#" + " "*20 + "FOOTBALL DECISION ENGINE" + " "*14 + "#")
    print("#" + " "*20 + "Demonstration Script" + " "*18 + "#")
    print("#"*60)

    # Run demos
    demo_elimination()
    demo_defensive_physics()
    demo_state_scoring()
    demo_defensive_blocks()
    demo_position_comparison()
    demo_visualization()

    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETE")
    print("="*60)
    print("\nThe Decision Engine provides:")
    print("  - Elimination as primary attacking metric")
    print("  - Physics-based defensive modeling")
    print("  - Composite state scoring")
    print("  - Block configuration templates")
    print("  - Tactical board visualization")
    print("\nSee DECISION_ENGINE_TECHNICAL_BRIEF.md for full documentation.")


if __name__ == "__main__":
    main()
