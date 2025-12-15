#!/usr/bin/env python3
"""
Modify G-code Z-hop behavior for ink extrusion.

Pattern detected:
1. Z increases (z-hop up)
2. Optional travel moves
3. Z decreases (z-hop down) to target height
4. Optional stationary extrusion (E without XY)
5. XY+E moves (extrude while moving)

Modification:
- Step 3: Z goes to (target + z_offset) instead of target
- Step 5+: Z linearly interpolates from (target + z_offset) to target
          over the specified transition distance (default 1mm of XY motion)
- If a single extrude move is longer than transition_distance, it is split
  into two segments: the first covers exactly transition_distance with Z
  interpolation, the second continues at target Z without modification.
  E (extrusion) values are split proportionally between segments.
"""

import math
import sys
from dataclasses import dataclass
from typing import Optional


def format_num(value: float) -> str:
    """Format number without scientific notation, trimming trailing zeros."""
    # Use enough decimal places for small values, then strip trailing zeros
    formatted = f"{value:.6f}".rstrip('0').rstrip('.')
    return formatted


@dataclass
class GCodeLine:
    """Parsed G-code line with components."""
    original: str
    command: Optional[str] = None  # G0, G1, etc.
    x: Optional[float] = None
    y: Optional[float] = None
    z: Optional[float] = None
    e: Optional[float] = None
    f: Optional[float] = None
    comment: str = ""

    def has_xy_move(self) -> bool:
        return self.x is not None or self.y is not None

    def has_e(self) -> bool:
        return self.e is not None

    def has_z(self) -> bool:
        return self.z is not None

    def is_extrude_move(self) -> bool:
        """XY movement with extrusion."""
        return self.has_xy_move() and self.has_e()

    def is_stationary_extrude(self) -> bool:
        """Extrusion without XY movement."""
        return self.has_e() and not self.has_xy_move() and not self.has_z()

    def rebuild(self, new_z: Optional[float] = None, add_z: Optional[float] = None,
                new_f: Optional[float] = None) -> str:
        """Rebuild the G-code line, optionally modifying Z and/or F."""
        if self.command is None:
            return self.original

        parts = [self.command]

        # Handle feedrate - use new_f if provided, otherwise original
        if new_f is not None:
            parts.append(f"F{format_num(new_f)}")
        elif self.f is not None:
            parts.append(f"F{format_num(self.f)}")
        if self.x is not None:
            parts.append(f"X{format_num(self.x)}")
        if self.y is not None:
            parts.append(f"Y{format_num(self.y)}")

        # Handle Z modification
        if new_z is not None:
            parts.append(f"Z{format_num(new_z)}")
        elif add_z is not None:
            parts.append(f"Z{format_num(add_z)}")
        elif self.z is not None:
            parts.append(f"Z{format_num(self.z)}")

        if self.e is not None:
            parts.append(f"E{format_num(self.e)}")

        result = " ".join(parts)
        if self.comment:
            result += " " + self.comment
        return result


def parse_gcode_line(line: str) -> GCodeLine:
    """Parse a G-code line into components."""
    original = line.rstrip('\n\r')
    stripped = original.strip()

    # Handle comments
    comment = ""
    if ';' in stripped:
        code_part, comment = stripped.split(';', 1)
        comment = ';' + comment
        stripped = code_part.strip()

    if not stripped:
        return GCodeLine(original=original, comment=comment)

    # Parse command and parameters
    tokens = stripped.split()
    if not tokens:
        return GCodeLine(original=original, comment=comment)

    command = tokens[0].upper()
    if command not in ('G0', 'G1', 'G2', 'G3'):
        return GCodeLine(original=original, command=command, comment=comment)

    result = GCodeLine(original=original, command=command, comment=comment)

    for token in tokens[1:]:
        if not token:
            continue
        param = token[0].upper()
        try:
            value = float(token[1:])
            if param == 'X':
                result.x = value
            elif param == 'Y':
                result.y = value
            elif param == 'Z':
                result.z = value
            elif param == 'E':
                result.e = value
            elif param == 'F':
                result.f = value
        except (ValueError, IndexError):
            pass

    return result


def process_gcode(input_path: str, output_path: str, z_offset: float = 0.2,
                   transition_distance: float = 1.0, min_hop_height: float = 2.0,
                   verbose: bool = False):
    """
    Process G-code file to modify Z-hop behavior.

    Args:
        input_path: Path to input G-code file
        output_path: Path to output G-code file
        z_offset: Amount to raise Z during initial descent (default 0.2mm)
        transition_distance: Distance over which to linearly transition Z (default 1.0mm)
        min_hop_height: Minimum hop height for travel clearance (default 2.0mm)
        verbose: Print debug information
    """
    with open(input_path, 'r') as f:
        lines = f.readlines()

    parsed_lines = [parse_gcode_line(line) for line in lines]
    output_lines = []

    current_z = 0.0
    previous_z = 0.0
    current_x = 0.0
    current_y = 0.0
    current_tool = None  # Track selected tool

    i = 0
    while i < len(parsed_lines):
        line = parsed_lines[i]

        # Track tool changes (T0, T1, T2, etc.)
        if line.command is not None and line.command.startswith('T') and line.command[1:].isdigit():
            current_tool = int(line.command[1:])
            if verbose:
                print(f"Line {i}: Tool changed to T{current_tool}")

        # Track position changes
        if line.has_z():
            previous_z = current_z
            current_z = line.z
        if line.x is not None:
            current_x = line.x
        if line.y is not None:
            current_y = line.y

        # Detect Z decrease (z-hop down) - only for T1
        # Skip very large Z drops (>10mm) as they're likely initial positioning, not hops
        if line.has_z() and line.z < previous_z and current_tool == 1 and (previous_z - line.z) < 10.0:
            target_z = line.z
            hop_height = previous_z - target_z  # How much the original hop was
            if verbose:
                print(f"Line {i}: Z-hop down detected (Z{previous_z} -> Z{target_z}), hop={hop_height:.3f}mm, tool=T{current_tool}")

            # If hop is smaller than min_hop_height, increase it for travel clearance
            # Find and modify the Z increase line in output_lines
            # Use small tolerance for floating point comparison
            if hop_height < min_hop_height - 0.001:
                # Need to increase the hop - find and modify ALL lines at the hop height
                new_hop_z = target_z + min_hop_height
                if verbose:
                    print(f"  Increasing hop from {hop_height:.3f}mm to {min_hop_height:.3f}mm (Z{previous_z} -> Z{new_hop_z})")
                for idx in range(len(output_lines) - 1, -1, -1):
                    check_line = parse_gcode_line(output_lines[idx])
                    if check_line.has_z() and check_line.z == previous_z:
                        # Found a line at hop height, increase it
                        output_lines[idx] = check_line.rebuild(new_z=new_hop_z)
                    elif check_line.has_z() and check_line.z < previous_z:
                        # Hit a Z below hop height - stop searching
                        break

            # Look ahead for the pattern:
            # - Optional stationary extrusion lines
            # - First line with XY + E (extrude while moving)

            j = i + 1
            first_extrude_move_idx = None

            while j < len(parsed_lines) and j < i + 15:  # Look up to 15 lines ahead
                ahead = parsed_lines[j]

                if ahead.is_extrude_move():
                    # Found the first extrude+move after Z decrease
                    first_extrude_move_idx = j
                    break
                elif ahead.is_stationary_extrude():
                    # Stationary extrusion - continue looking
                    j += 1
                elif ahead.has_z():
                    # Another Z move - pattern broken
                    break
                elif ahead.has_xy_move() and not ahead.has_e():
                    # XY move without extrusion - pattern broken
                    break
                elif ahead.command in ('G0', 'G1') and not ahead.has_xy_move() and not ahead.has_e() and not ahead.has_z():
                    # Just feedrate change or empty move - continue
                    j += 1
                elif ahead.command is None:
                    # Comment or empty line - continue
                    j += 1
                else:
                    # Other command - might break pattern
                    break

            if first_extrude_move_idx is not None:
                # Pattern found! Modify the lines
                # Hop has been ensured to be at least min_hop_height above
                # Now drop to target + z_offset before smoothing

                drop_z = target_z + z_offset
                # Preserve feedrate from original line if present
                if line.f is not None:
                    output_lines.append(f"G1 F{format_num(line.f)} Z{format_num(drop_z)}")
                else:
                    output_lines.append(f"G1 Z{format_num(drop_z)}")

                # Copy intermediate lines unchanged
                for k in range(i + 1, first_extrude_move_idx):
                    output_lines.append(parsed_lines[k].original)

                # Process extrude+move lines with Z interpolation based on XY distance
                cumulative_distance = 0.0
                k = first_extrude_move_idx
                # Track position for distance calculation
                interp_x = current_x
                interp_y = current_y
                # Track feedrate for restoration after smoothing
                original_feedrate = None
                # Track E value for proper splitting (assumes absolute E coordinates)
                previous_e = 0.0
                # Get previous E from intermediate lines if any stationary extrusion occurred
                for m in range(i + 1, first_extrude_move_idx):
                    if parsed_lines[m].has_e():
                        previous_e = parsed_lines[m].e

                while k < len(parsed_lines):
                    extrude_line = parsed_lines[k]

                    if extrude_line.is_extrude_move():
                        # Track original feedrate from first move with F
                        if original_feedrate is None and extrude_line.f is not None:
                            original_feedrate = extrude_line.f

                        # Calculate XY distance traveled
                        new_x = extrude_line.x if extrude_line.x is not None else interp_x
                        new_y = extrude_line.y if extrude_line.y is not None else interp_y
                        dx = new_x - interp_x
                        dy = new_y - interp_y
                        distance = math.sqrt(dx * dx + dy * dy)

                        # Check if this move would exceed transition distance
                        # If so, split the move at exactly the transition point
                        if cumulative_distance < transition_distance and cumulative_distance + distance > transition_distance:
                            # This move crosses the transition boundary - split it
                            remaining_transition = transition_distance - cumulative_distance
                            split_fraction = remaining_transition / distance if distance > 0 else 1.0

                            # Calculate intermediate point
                            split_x = interp_x + dx * split_fraction
                            split_y = interp_y + dy * split_fraction

                            # Calculate proportional extrusion for the first segment
                            # E is absolute, so split_e = previous_e + (target_e - previous_e) * fraction
                            split_e = None
                            if extrude_line.e is not None:
                                e_delta = extrude_line.e - previous_e
                                split_e = previous_e + e_delta * split_fraction

                            # First segment: from current position to split point, with Z = target_z
                            # Increase feedrate by 50% during smoothing
                            boosted_f = extrude_line.f * 1.5 if extrude_line.f is not None else None
                            first_parts = ["G1"]
                            if boosted_f is not None:
                                first_parts.append(f"F{format_num(boosted_f)}")
                            first_parts.append(f"X{format_num(split_x)}")
                            first_parts.append(f"Y{format_num(split_y)}")
                            first_parts.append(f"Z{format_num(target_z)}")  # Transition complete at split
                            if split_e is not None:
                                first_parts.append(f"E{format_num(split_e)}")
                            output_lines.append(" ".join(first_parts))

                            # Restore original feedrate after smoothing segment
                            if original_feedrate is not None:
                                output_lines.append(f"G1 F{format_num(original_feedrate)}")

                            # Second segment: from split point to destination, no Z modification
                            second_parts = ["G1"]
                            if extrude_line.f is not None and original_feedrate is None:
                                second_parts.append(f"F{format_num(extrude_line.f)}")
                            second_parts.append(f"X{format_num(new_x)}")
                            second_parts.append(f"Y{format_num(new_y)}")
                            if extrude_line.e is not None:
                                second_parts.append(f"E{format_num(extrude_line.e)}")
                            if extrude_line.comment:
                                second_parts.append(extrude_line.comment)
                            output_lines.append(" ".join(second_parts))

                            # Update tracking
                            interp_x, interp_y = new_x, new_y
                            cumulative_distance = transition_distance  # Mark transition as complete
                            k += 1
                            break  # Transition complete, exit loop

                        cumulative_distance += distance
                        interp_x, interp_y = new_x, new_y
                        # Update previous_e for next iteration
                        if extrude_line.e is not None:
                            previous_e = extrude_line.e

                        # Calculate Z based on how far through transition we are
                        # Z transitions from (target+offset) to target over transition_distance
                        if cumulative_distance >= transition_distance:
                            # Transition complete - use target Z
                            interpolated_z = target_z
                        else:
                            # Interpolate Z based on progress
                            progress = cumulative_distance / transition_distance
                            interpolated_z = target_z + z_offset * (1.0 - progress)

                        # Increase feedrate by 50% during smoothing
                        boosted_f = extrude_line.f * 1.5 if extrude_line.f is not None else None
                        output_lines.append(extrude_line.rebuild(add_z=interpolated_z, new_f=boosted_f))
                        k += 1

                        # Stop adding Z after transition is complete
                        if cumulative_distance >= transition_distance:
                            # Restore original feedrate after smoothing
                            if original_feedrate is not None:
                                output_lines.append(f"G1 F{format_num(original_feedrate)}")
                            break
                    elif extrude_line.is_stationary_extrude():
                        # Stationary extrusion during transition - keep unchanged
                        output_lines.append(extrude_line.original)
                        k += 1
                    elif extrude_line.has_z():
                        # Another Z move - stop interpolation
                        break
                    elif extrude_line.command is None:
                        # Comment or empty line - keep and continue
                        output_lines.append(extrude_line.original)
                        k += 1
                    else:
                        # Other command - stop interpolation
                        break

                # Update tracked position
                current_x, current_y = interp_x, interp_y

                # Update current_z to target
                current_z = target_z

                # Skip to after the modified section
                i = k
                continue

        # No pattern match - output line unchanged
        output_lines.append(line.original)
        i += 1

    with open(output_path, 'w') as f:
        for line in output_lines:
            f.write(line + '\n')

    print(f"Processed {input_path} -> {output_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python ink_z_starts.py <input.gcode> [output.gcode] [z_offset] [transition_dist] [min_hop] [-v]")
        print("  input.gcode     - Input G-code file")
        print("  output.gcode    - Output file (default: input_modified.gcode)")
        print("  z_offset        - Z offset for smoothing start (default: 0.2mm)")
        print("  transition_dist - XY distance to transition Z over (default: 1.0mm)")
        print("  min_hop         - Minimum hop height for travel clearance (default: 2.0mm)")
        print("  -v              - Verbose mode (show debug info)")
        sys.exit(1)

    # Check for verbose flag
    verbose = '-v' in sys.argv
    args = [a for a in sys.argv[1:] if a != '-v']

    input_path = args[0]

    if len(args) >= 2:
        output_path = args[1]
    else:
        # Default output name
        if input_path.endswith('.gcode'):
            output_path = input_path[:-6] + '_modified.gcode'
        else:
            output_path = input_path + '_modified.gcode'

    z_offset = 0.2
    if len(args) >= 3:
        z_offset = float(args[2])

    transition_distance = 1.0
    if len(args) >= 4:
        transition_distance = float(args[3])

    min_hop_height = 2.0
    if len(args) >= 5:
        min_hop_height = float(args[4])

    process_gcode(input_path, output_path, z_offset, transition_distance, min_hop_height, verbose)


if __name__ == "__main__":
    main()
