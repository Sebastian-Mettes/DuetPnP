#!/usr/bin/env python3
"""
Modify G-code Z-hop behavior for ink extrusion.

Pattern detected:
1. Z increases (z-hop up)
2. Optional travel moves
3. Z decreases (z-hop down) to target height
4. Optional stationary extrusion (E without XY)
5. First XY+E move (extrude while moving)

Modification:
- Step 3: Z goes to (target + 0.2) instead of target
- Step 5: Add original target Z to the move command
"""

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

    def rebuild(self, new_z: Optional[float] = None, add_z: Optional[float] = None) -> str:
        """Rebuild the G-code line, optionally modifying Z."""
        if self.command is None:
            return self.original

        parts = [self.command]

        if self.f is not None:
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


def process_gcode(input_path: str, output_path: str, z_offset: float = 0.2):
    """
    Process G-code file to modify Z-hop behavior.

    Args:
        input_path: Path to input G-code file
        output_path: Path to output G-code file
        z_offset: Amount to raise Z during initial descent (default 0.2mm)
    """
    with open(input_path, 'r') as f:
        lines = f.readlines()

    parsed_lines = [parse_gcode_line(line) for line in lines]
    output_lines = []

    current_z = 0.0
    previous_z = 0.0

    i = 0
    while i < len(parsed_lines):
        line = parsed_lines[i]

        # Track Z position changes
        if line.has_z():
            previous_z = current_z
            current_z = line.z

        # Detect Z decrease (z-hop down)
        if line.has_z() and line.z < previous_z:
            target_z = line.z

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

                # Modify Z decrease line: raise by offset
                modified_z = target_z + z_offset
                output_lines.append(line.rebuild(new_z=modified_z))

                # Copy intermediate lines unchanged
                for k in range(i + 1, first_extrude_move_idx):
                    output_lines.append(parsed_lines[k].original)

                # Modify extrude+move line: add original target Z
                extrude_line = parsed_lines[first_extrude_move_idx]
                output_lines.append(extrude_line.rebuild(add_z=target_z))

                # Update current_z to target (since we're adding it to the move)
                current_z = target_z

                # Skip to after the modified section
                i = first_extrude_move_idx + 1
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
        print("Usage: python ink_z_starts.py <input.gcode> [output.gcode] [z_offset]")
        print("  input.gcode  - Input G-code file")
        print("  output.gcode - Output file (default: input_modified.gcode)")
        print("  z_offset     - Z offset in mm (default: 0.2)")
        sys.exit(1)

    input_path = sys.argv[1]

    if len(sys.argv) >= 3:
        output_path = sys.argv[2]
    else:
        # Default output name
        if input_path.endswith('.gcode'):
            output_path = input_path[:-6] + '_modified.gcode'
        else:
            output_path = input_path + '_modified.gcode'

    z_offset = 0.2
    if len(sys.argv) >= 4:
        z_offset = float(sys.argv[3])

    process_gcode(input_path, output_path, z_offset)


if __name__ == "__main__":
    main()
