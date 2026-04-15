"""
Unit tests for Utility/angle_calculations.py

Tests cover:
  - calculate_angle        (three-point joint angle)
  - angle_to_vertical_degrees
  - thigh_to_horizontal
"""

import sys
import os
import math
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from Utility.angle_calculations import (
    calculate_angle,
    angle_to_vertical_degrees,
    thigh_to_horizontal,
)


# ─────────────────────────────────────────────────────────────
# calculate_angle
# ─────────────────────────────────────────────────────────────

class TestCalculateAngle:

    def test_right_angle(self):
        """Classic L-shape at the origin: angle at b should be 90°."""
        a = [1, 0]
        b = [0, 0]
        c = [0, 1]
        assert abs(calculate_angle(a, b, c) - 90.0) < 0.01

    def test_straight_line_180(self):
        """Three collinear points — angle at b should be 180°."""
        a = [1, 0]
        b = [0, 0]
        c = [-1, 0]
        assert abs(calculate_angle(a, b, c) - 180.0) < 0.01

    def test_45_degree_angle(self):
        a = [1, 0]
        b = [0, 0]
        c = [1, 1]
        assert abs(calculate_angle(a, b, c) - 45.0) < 0.01

    def test_60_degree_angle(self):
        """Equilateral triangle corner — 60°."""
        a = [1, 0]
        b = [0, 0]
        c = [0.5, math.sqrt(3) / 2]
        assert abs(calculate_angle(a, b, c) - 60.0) < 0.1

    def test_symmetry(self):
        """Swapping a and c should give the same angle."""
        a = [2, 1]
        b = [0, 0]
        c = [1, 3]
        assert abs(calculate_angle(a, b, c) - calculate_angle(c, b, a)) < 0.01

    def test_does_not_return_nan(self):
        """Floating-point edge case: nearly collinear vectors should not produce NaN."""
        a = [1.0, 0.0]
        b = [0.0, 0.0]
        c = [-1.0 + 1e-10, 1e-10]
        result = calculate_angle(a, b, c)
        assert not math.isnan(result)
        assert 0 <= result <= 180

    def test_normalised_coords_knee_angle(self):
        """Simulate a 90° knee angle using normalised MediaPipe-style coords."""
        hip   = [0.5, 0.3]
        knee  = [0.5, 0.6]
        ankle = [0.8, 0.6]
        angle = calculate_angle(hip, knee, ankle)
        assert abs(angle - 90.0) < 1.0


# ─────────────────────────────────────────────────────────────
# angle_to_vertical_degrees
# ─────────────────────────────────────────────────────────────

class TestAngleToVertical:

    def test_perfectly_vertical_downward(self):
        """Vector pointing straight down → 0°."""
        assert abs(angle_to_vertical_degrees([0, 0], [0, 1])) < 0.01

    def test_perfectly_vertical_upward(self):
        """Vector pointing straight up → 0°."""
        assert abs(angle_to_vertical_degrees([0, 1], [0, 0])) < 0.01

    def test_perfectly_horizontal(self):
        """Horizontal vector → 90°."""
        assert abs(angle_to_vertical_degrees([0, 0], [1, 0]) - 90.0) < 0.01

    def test_45_degree_diagonal(self):
        assert abs(angle_to_vertical_degrees([0, 0], [1, 1]) - 45.0) < 0.01

    def test_same_points_returns_zero(self):
        """Identical points should not crash and should return 0."""
        result = angle_to_vertical_degrees([0.5, 0.5], [0.5, 0.5])
        assert result == 0.0

    def test_torso_lean_upright(self):
        """
        Hip directly below shoulder in normalised coords.
        hip=(0.5,0.7), shoulder=(0.5,0.2) → vector is straight up → 0°.
        """
        hip      = [0.5, 0.7]
        shoulder = [0.5, 0.2]
        angle = angle_to_vertical_degrees(hip, shoulder)
        assert angle < 5.0  # near vertical

    def test_torso_lean_45(self):
        """Hip to shoulder vector at 45° from vertical."""
        hip      = [0.5, 0.7]
        shoulder = [0.0, 0.2]   # dx = -0.5, dy = -0.5  → 45°
        angle = angle_to_vertical_degrees(hip, shoulder)
        assert abs(angle - 45.0) < 1.0


# ─────────────────────────────────────────────────────────────
# thigh_to_horizontal
# ─────────────────────────────────────────────────────────────

class TestThighToHorizontal:

    def test_standing_vertical_thigh(self):
        """Hip directly above knee → thigh is vertical → 0°."""
        hip  = [0.5, 0.3]
        knee = [0.5, 0.6]
        assert abs(thigh_to_horizontal(hip, knee)) < 0.01

    def test_parallel_to_ground(self):
        """Hip and knee at same height → thigh is horizontal → 90°."""
        hip  = [0.3, 0.5]
        knee = [0.6, 0.5]
        assert abs(thigh_to_horizontal(hip, knee) - 90.0) < 0.01

    def test_45_degree_thigh(self):
        hip  = [0.5, 0.3]
        knee = [0.6, 0.4]   # equal dx and dy
        assert abs(thigh_to_horizontal(hip, knee) - 45.0) < 0.5

    def test_deep_squat_above_90(self):
        """
        In a very deep squat the knee drops below and forward of the hip,
        giving a thigh angle above 90° in some configurations.
        Result should still be a valid float between 0 and 180.
        """
        hip  = [0.5, 0.4]
        knee = [0.65, 0.5]
        result = thigh_to_horizontal(hip, knee)
        assert 0 <= result <= 180