"""
Decision Engine Visualization Layer

All outputs visualized on a tactical board, not as raw numbers.

Visualization goals:
- Full pitch, top-down
- 22 players + ball
- Zones clearly visible
- Ability to see:
  - Defensive block shape
  - Lines of pressure
  - Eliminations as they occur
  - Changes when ball moves

This is essential for coach adoption. If it cannot be read
like a tactics board, it fails regardless of model quality.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple, Any
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch, Circle, Polygon, FancyArrowPatch
    from matplotlib.collections import PatchCollection
    from matplotlib.colors import LinearSegmentedColormap
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from mplsoccer import Pitch
    HAS_MPLSOCCER = True
except ImportError:
    HAS_MPLSOCCER = False

from .pitch_geometry import (
    Position,
    PitchGeometry,
    HALF_LENGTH,
    HALF_WIDTH,
    PITCH_LENGTH,
    PITCH_WIDTH,
    GOAL_WIDTH,
)
from .elimination import (
    Player,
    EliminationState,
    EliminationResult,
    DefenderStatus,
)
from .defense_physics import (
    DefensiveShape,
    AttractionForce,
    CoverShadowCalculator,
)
from .state_scoring import GameState, StateScore
from .block_models import DefensiveBlock, BlockType


@dataclass
class VisualizationConfig:
    """Configuration for visualization appearance."""
    # Colors
    attack_color: str = "#E74C3C"       # Red for attacking team
    defense_color: str = "#3498DB"       # Blue for defending team
    ball_color: str = "#F1C40F"          # Yellow for ball
    eliminated_color: str = "#95A5A6"    # Gray for eliminated defenders
    active_color: str = "#2ECC71"        # Green for active defenders

    # Sizes
    player_size: float = 300
    ball_size: float = 200
    line_width: float = 2

    # Pitch
    pitch_color: str = "#1a472a"         # Dark green
    line_color: str = "white"

    # Labels
    show_player_ids: bool = True
    show_elimination_status: bool = True
    show_force_vectors: bool = False
    show_cover_shadows: bool = True

    # Heatmap
    heatmap_cmap: str = "RdYlGn"
    heatmap_alpha: float = 0.6


class DecisionEngineVisualizer:
    """
    Visualizer for the decision engine.

    Creates tactical board visualizations that coaches can read.
    """

    def __init__(
        self,
        config: Optional[VisualizationConfig] = None,
        geometry: Optional[PitchGeometry] = None,
        use_mplsoccer: bool = True,
    ):
        if not HAS_MATPLOTLIB:
            raise ImportError(
                "matplotlib is required for visualization. "
                "Install with: pip install matplotlib"
            )

        self.config = config or VisualizationConfig()
        self.geometry = geometry or PitchGeometry()
        self.use_mplsoccer = use_mplsoccer and HAS_MPLSOCCER
        self.shadow_calc = CoverShadowCalculator(geometry=self.geometry)

    def _create_pitch(
        self,
        figsize: Tuple[int, int] = (14, 9),
    ) -> Tuple[Any, Any]:
        """Create a pitch figure."""
        if self.use_mplsoccer:
            # Use mplsoccer for professional pitch rendering
            pitch = Pitch(
                pitch_type='custom',
                pitch_length=PITCH_LENGTH,
                pitch_width=PITCH_WIDTH,
                pitch_color=self.config.pitch_color,
                line_color=self.config.line_color,
                linewidth=2,
            )
            fig, ax = pitch.draw(figsize=figsize)

            # Adjust to use our coordinate system (origin at center)
            # mplsoccer uses (0,0) at corner, we need to offset
            return fig, ax
        else:
            # Manual pitch drawing
            fig, ax = plt.subplots(figsize=figsize)
            self._draw_pitch_manual(ax)
            return fig, ax

    def _draw_pitch_manual(self, ax: Any) -> None:
        """Draw pitch manually without mplsoccer."""
        ax.set_facecolor(self.config.pitch_color)

        # Pitch outline
        ax.plot(
            [-HALF_LENGTH, HALF_LENGTH, HALF_LENGTH, -HALF_LENGTH, -HALF_LENGTH],
            [-HALF_WIDTH, -HALF_WIDTH, HALF_WIDTH, HALF_WIDTH, -HALF_WIDTH],
            color=self.config.line_color,
            linewidth=self.config.line_width,
        )

        # Center line
        ax.axvline(x=0, color=self.config.line_color, linewidth=self.config.line_width)

        # Center circle
        circle = plt.Circle((0, 0), 9.15, fill=False,
                           color=self.config.line_color,
                           linewidth=self.config.line_width)
        ax.add_patch(circle)

        # Penalty areas
        for sign in [-1, 1]:
            # Penalty area
            ax.plot(
                [sign * HALF_LENGTH, sign * (HALF_LENGTH - 16.5),
                 sign * (HALF_LENGTH - 16.5), sign * HALF_LENGTH],
                [-20.16, -20.16, 20.16, 20.16],
                color=self.config.line_color,
                linewidth=self.config.line_width,
            )

            # Goal area
            ax.plot(
                [sign * HALF_LENGTH, sign * (HALF_LENGTH - 5.5),
                 sign * (HALF_LENGTH - 5.5), sign * HALF_LENGTH],
                [-9.16, -9.16, 9.16, 9.16],
                color=self.config.line_color,
                linewidth=self.config.line_width,
            )

            # Goal
            ax.plot(
                [sign * HALF_LENGTH, sign * (HALF_LENGTH + 1),
                 sign * (HALF_LENGTH + 1), sign * HALF_LENGTH],
                [-GOAL_WIDTH/2, -GOAL_WIDTH/2, GOAL_WIDTH/2, GOAL_WIDTH/2],
                color=self.config.line_color,
                linewidth=self.config.line_width * 1.5,
            )

        ax.set_xlim(-HALF_LENGTH - 5, HALF_LENGTH + 5)
        ax.set_ylim(-HALF_WIDTH - 3, HALF_WIDTH + 3)
        ax.set_aspect('equal')
        ax.axis('off')

    def _convert_coords(self, pos: Position) -> Tuple[float, float]:
        """
        Convert our coordinate system to visualization coordinates.

        Our system: origin at center, +x toward attacking goal
        mplsoccer: origin at corner (0,0), different orientation
        """
        if self.use_mplsoccer:
            # Convert from centered to corner origin
            x = pos.x + HALF_LENGTH
            y = pos.y + HALF_WIDTH
            return x, y
        else:
            return pos.x, pos.y

    def plot_game_state(
        self,
        state: GameState,
        title: str = "Game State",
        show_score: bool = True,
        figsize: Tuple[int, int] = (14, 9),
    ) -> Any:
        """
        Plot a complete game state.

        Shows all players, ball, and annotations.
        """
        fig, ax = self._create_pitch(figsize=figsize)

        # Plot ball
        ball_x, ball_y = self._convert_coords(state.ball_position)
        ax.scatter(
            [ball_x], [ball_y],
            c=self.config.ball_color,
            s=self.config.ball_size,
            zorder=10,
            edgecolors='black',
            linewidths=2,
            marker='o',
        )

        # Plot attackers
        for attacker in state.attackers:
            x, y = self._convert_coords(attacker.position)
            ax.scatter(
                [x], [y],
                c=self.config.attack_color,
                s=self.config.player_size,
                zorder=5,
                edgecolors='white',
                linewidths=1.5,
                marker='o',
            )
            if self.config.show_player_ids:
                ax.annotate(
                    attacker.id,
                    (x, y),
                    ha='center',
                    va='center',
                    fontsize=8,
                    fontweight='bold',
                    color='white',
                    zorder=6,
                )

        # Plot defenders with elimination status
        if state.elimination_state:
            for result in state.elimination_state.defenders:
                self._plot_defender_with_status(ax, result)
        else:
            for defender in state.defenders:
                x, y = self._convert_coords(defender.position)
                ax.scatter(
                    [x], [y],
                    c=self.config.defense_color,
                    s=self.config.player_size,
                    zorder=5,
                    edgecolors='white',
                    linewidths=1.5,
                )

        # Add cover shadows if enabled
        if self.config.show_cover_shadows and state.defenders:
            self._plot_cover_shadows(ax, state.defenders, state.ball_position)

        # Add score display
        if show_score and state.score:
            self._add_score_panel(ax, state.score)

        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

        return fig

    def _plot_defender_with_status(
        self,
        ax: Any,
        result: EliminationResult,
    ) -> None:
        """Plot a defender with elimination status indication."""
        x, y = self._convert_coords(result.defender.position)

        # Color based on status
        if result.status == DefenderStatus.ELIMINATED:
            color = self.config.eliminated_color
            marker = 'o'
            alpha = 0.5
        elif result.status == DefenderStatus.COVERING:
            color = self.config.active_color
            marker = 'o'
            alpha = 1.0
        else:
            color = self.config.defense_color
            marker = 'o'
            alpha = 0.8

        ax.scatter(
            [x], [y],
            c=color,
            s=self.config.player_size,
            zorder=5,
            edgecolors='white',
            linewidths=1.5,
            marker=marker,
            alpha=alpha,
        )

        # Add ID and status
        if self.config.show_player_ids:
            ax.annotate(
                result.defender.id,
                (x, y),
                ha='center',
                va='center',
                fontsize=8,
                fontweight='bold',
                color='white' if alpha == 1.0 else 'black',
                zorder=6,
            )

        # Show intervention line for eliminated defenders
        if self.config.show_elimination_status and result.is_eliminated:
            int_x, int_y = self._convert_coords(result.intervention_point)
            ax.plot(
                [x, int_x], [y, int_y],
                color=self.config.eliminated_color,
                linestyle='--',
                linewidth=1,
                alpha=0.5,
            )

    def _plot_cover_shadows(
        self,
        ax: Any,
        defenders: List[Player],
        ball_position: Position,
    ) -> None:
        """Plot cover shadows for all defenders."""
        for defender in defenders:
            apex, left, right = self.shadow_calc.calculate_shadow(
                defender.position,
                ball_position,
            )

            # Convert coordinates
            apex_xy = self._convert_coords(apex)
            left_xy = self._convert_coords(left)
            right_xy = self._convert_coords(right)

            # Draw shadow triangle
            triangle = Polygon(
                [apex_xy, left_xy, right_xy],
                alpha=0.15,
                facecolor=self.config.defense_color,
                edgecolor='none',
            )
            ax.add_patch(triangle)

    def _add_score_panel(self, ax: Any, score: StateScore) -> None:
        """Add a panel showing the state score breakdown."""
        text_lines = [
            f"Total Score: {score.total:.2f}",
            f"Elimination: {score.elimination_score:.2f}",
            f"Proximity: {score.proximity_score:.2f}",
            f"Angle: {score.angle_score:.2f}",
            f"Density: {score.density_score:.2f}",
        ]

        text = "\n".join(text_lines)

        # Position in top-right corner
        if self.use_mplsoccer:
            x_pos = PITCH_LENGTH - 5
            y_pos = PITCH_WIDTH - 5
        else:
            x_pos = HALF_LENGTH - 5
            y_pos = HALF_WIDTH - 5

        ax.text(
            x_pos, y_pos,
            text,
            fontsize=9,
            fontfamily='monospace',
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(
                boxstyle='round',
                facecolor='white',
                alpha=0.8,
            ),
        )

    def plot_elimination_state(
        self,
        state: EliminationState,
        attackers: Optional[List[Player]] = None,
        title: str = "Elimination Analysis",
        figsize: Tuple[int, int] = (14, 9),
    ) -> Any:
        """
        Plot elimination state with detailed annotations.

        Shows which defenders are eliminated and why.
        """
        fig, ax = self._create_pitch(figsize=figsize)

        # Plot ball
        ball_x, ball_y = self._convert_coords(state.ball_position)
        ax.scatter(
            [ball_x], [ball_y],
            c=self.config.ball_color,
            s=self.config.ball_size,
            zorder=10,
            edgecolors='black',
            linewidths=2,
        )

        # Plot ball carrier
        if state.ball_carrier:
            x, y = self._convert_coords(state.ball_carrier.position)
            ax.scatter(
                [x], [y],
                c=self.config.attack_color,
                s=self.config.player_size * 1.2,
                zorder=9,
                edgecolors='gold',
                linewidths=3,
            )

        # Plot attackers
        if attackers:
            for attacker in attackers:
                if state.ball_carrier and attacker.id == state.ball_carrier.id:
                    continue
                x, y = self._convert_coords(attacker.position)
                ax.scatter(
                    [x], [y],
                    c=self.config.attack_color,
                    s=self.config.player_size,
                    zorder=5,
                    edgecolors='white',
                    linewidths=1.5,
                )

        # Plot defenders with detailed status
        for result in state.defenders:
            self._plot_defender_detailed(ax, result, state.ball_position)

        # Add ball-to-goal line
        goal_x, goal_y = self._convert_coords(self.geometry.attacking_goal)
        ax.plot(
            [ball_x, goal_x], [ball_y, goal_y],
            color='white',
            linestyle=':',
            linewidth=2,
            alpha=0.5,
            zorder=1,
        )

        # Add legend
        self._add_elimination_legend(ax)

        # Add stats
        stats_text = (
            f"Eliminated: {state.eliminated_count}/{len(state.defenders)}\n"
            f"Ratio: {state.elimination_ratio:.1%}"
        )
        ax.text(
            0.02, 0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=11,
            fontweight='bold',
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
        )

        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

        return fig

    def _plot_defender_detailed(
        self,
        ax: Any,
        result: EliminationResult,
        ball_position: Position,
    ) -> None:
        """Plot defender with detailed elimination information."""
        x, y = self._convert_coords(result.defender.position)
        int_x, int_y = self._convert_coords(result.intervention_point)

        # Base color
        if result.is_eliminated:
            color = self.config.eliminated_color
            edgecolor = 'red'
        else:
            color = self.config.defense_color
            edgecolor = 'white'

        # Plot player
        ax.scatter(
            [x], [y],
            c=color,
            s=self.config.player_size,
            zorder=5,
            edgecolors=edgecolor,
            linewidths=2,
        )

        # Plot intervention point
        ax.scatter(
            [int_x], [int_y],
            c='white',
            s=50,
            zorder=4,
            marker='x',
            linewidths=2,
        )

        # Line to intervention point
        ax.plot(
            [x, int_x], [y, int_y],
            color=color,
            linestyle='--',
            linewidth=1,
            alpha=0.5,
        )

        # Time annotation
        time_text = f"{result.time_to_intervene:.1f}s"
        ax.annotate(
            time_text,
            ((x + int_x) / 2, (y + int_y) / 2),
            fontsize=7,
            color='white',
            ha='center',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.5),
        )

    def _add_elimination_legend(self, ax: Any) -> None:
        """Add legend for elimination visualization."""
        legend_elements = [
            plt.scatter([], [], c=self.config.defense_color, s=100,
                       edgecolors='white', linewidths=1.5, label='Active'),
            plt.scatter([], [], c=self.config.eliminated_color, s=100,
                       edgecolors='red', linewidths=2, label='Eliminated'),
            plt.scatter([], [], c=self.config.active_color, s=100,
                       edgecolors='white', linewidths=1.5, label='Covering'),
        ]
        ax.legend(
            handles=legend_elements,
            loc='upper left',
            framealpha=0.9,
        )

    def plot_defensive_block(
        self,
        block: DefensiveBlock,
        ball_position: Position,
        formation: str = "4-4-2",
        title: str = "Defensive Block",
        figsize: Tuple[int, int] = (14, 9),
    ) -> Any:
        """
        Visualize a defensive block configuration.
        """
        fig, ax = self._create_pitch(figsize=figsize)

        # Calculate block positions
        positions = block.calculate_positions(ball_position, formation)

        # Plot ball
        ball_x, ball_y = self._convert_coords(ball_position)
        ax.scatter(
            [ball_x], [ball_y],
            c=self.config.ball_color,
            s=self.config.ball_size,
            zorder=10,
            edgecolors='black',
            linewidths=2,
        )

        # Plot each position
        for pos_name, pos in positions.items():
            x, y = self._convert_coords(pos)
            ax.scatter(
                [x], [y],
                c=self.config.defense_color,
                s=self.config.player_size,
                zorder=5,
                edgecolors='white',
                linewidths=1.5,
            )
            ax.annotate(
                pos_name.split('_')[0][0].upper(),
                (x, y),
                ha='center',
                va='center',
                fontsize=8,
                fontweight='bold',
                color='white',
            )

        # Draw block lines
        line_heights = block.config.get_line_heights()
        for height in line_heights:
            line_pos = Position(height, 0)
            lx, _ = self._convert_coords(line_pos)
            if self.use_mplsoccer:
                ax.axvline(x=lx, color='white', linestyle=':', alpha=0.3)
            else:
                ax.axvline(x=height, color='white', linestyle=':', alpha=0.3)

        # Add block info
        info_text = (
            f"Block Type: {block.block_type.value.upper()}\n"
            f"Formation: {formation}\n"
            f"Def Line: {block.config.defensive_line_height:.0f}m\n"
            f"Press Trigger: {block.config.press_trigger_distance:.0f}m"
        )
        ax.text(
            0.02, 0.02,
            info_text,
            transform=ax.transAxes,
            fontsize=9,
            fontfamily='monospace',
            verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
        )

        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

        return fig

    def plot_value_heatmap(
        self,
        heatmap: np.ndarray,
        title: str = "Position Value Heatmap",
        figsize: Tuple[int, int] = (14, 9),
    ) -> Any:
        """
        Plot a heatmap of position values across the pitch.
        """
        fig, ax = self._create_pitch(figsize=figsize)

        # Create custom colormap
        cmap = plt.get_cmap(self.config.heatmap_cmap)

        # Overlay heatmap
        if self.use_mplsoccer:
            extent = [0, PITCH_LENGTH, 0, PITCH_WIDTH]
        else:
            extent = [-HALF_LENGTH, HALF_LENGTH, -HALF_WIDTH, HALF_WIDTH]

        im = ax.imshow(
            heatmap,
            extent=extent,
            origin='lower',
            cmap=cmap,
            alpha=self.config.heatmap_alpha,
            zorder=0,
            aspect='auto',
        )

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.6, label='Position Value')

        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

        return fig

    def plot_action_options(
        self,
        state: GameState,
        title: str = "Available Actions",
        top_n: int = 5,
        figsize: Tuple[int, int] = (14, 9),
    ) -> Any:
        """
        Visualize available actions and their expected values.
        """
        fig, ax = self._create_pitch(figsize=figsize)

        # Plot base state
        ball_x, ball_y = self._convert_coords(state.ball_position)
        ax.scatter(
            [ball_x], [ball_y],
            c=self.config.ball_color,
            s=self.config.ball_size,
            zorder=10,
            edgecolors='black',
            linewidths=2,
        )

        # Plot players
        for attacker in state.attackers:
            x, y = self._convert_coords(attacker.position)
            ax.scatter([x], [y], c=self.config.attack_color, s=self.config.player_size,
                      zorder=5, edgecolors='white', linewidths=1.5)

        for defender in state.defenders:
            x, y = self._convert_coords(defender.position)
            ax.scatter([x], [y], c=self.config.defense_color, s=self.config.player_size,
                      zorder=5, edgecolors='white', linewidths=1.5)

        # Plot top actions
        for i, action in enumerate(state.available_actions[:top_n]):
            target_x, target_y = self._convert_coords(action.target)

            # Color by expected value (green = good, red = bad)
            ev_normalized = (action.expected_value + 0.5) / 1.0  # Normalize to 0-1
            ev_normalized = max(0, min(1, ev_normalized))
            color = plt.cm.RdYlGn(ev_normalized)

            # Draw arrow
            arrow = FancyArrowPatch(
                (ball_x, ball_y),
                (target_x, target_y),
                arrowstyle='-|>',
                mutation_scale=15,
                linewidth=2,
                color=color,
                alpha=0.8,
                zorder=8,
            )
            ax.add_patch(arrow)

            # Label
            label = f"{action.action_type.value}: {action.expected_value:.2f}"
            mid_x = (ball_x + target_x) / 2
            mid_y = (ball_y + target_y) / 2
            ax.annotate(
                label,
                (mid_x, mid_y),
                fontsize=7,
                color='black',
                ha='center',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7),
            )

        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

        return fig

    def create_animation_frames(
        self,
        states: List[GameState],
        output_dir: str,
        prefix: str = "frame",
    ) -> List[str]:
        """
        Create a series of frames for animation.

        Returns list of saved file paths.
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        paths = []
        for i, state in enumerate(states):
            fig = self.plot_game_state(
                state,
                title=f"Frame {i+1}",
                show_score=True,
            )
            path = os.path.join(output_dir, f"{prefix}_{i:04d}.png")
            fig.savefig(path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            paths.append(path)

        return paths

    def save_figure(
        self,
        fig: Any,
        path: str,
        dpi: int = 150,
    ) -> None:
        """Save a figure to file."""
        fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
