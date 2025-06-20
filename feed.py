import calibration

class Feeder:
    def __init__(self, num_belts = 3, radius = 22.5, control_axis = 'B'):
        if not isinstance(self.num_belts, int):
            raise TypeError("num_belts must be an integer")
        if self.num_belts < 1:
            raise ValueError("num_belts must be greater than 0")
            
        if not isinstance(self.radius, (int, float)):
            raise TypeError("radius must be a number")
        if self.radius <= 0:
            raise ValueError("radius must be greater than 0")
        self.radius = radius
        self.num_belts = num_belts
        self.belt = None
    
    def home(self):
        """
        Home the feeder by rotating until stall, then allow user to test feed positions.
        User identifies which belt number (0-N from left) successfully feeds.
        """
        cal = calibration.CalibrateToolheads()
        
        # Slowly rotate until stall
        print("Rotating feeder until stall...")
        cal.send_gcode_command("G1 B120 F900")  # Rotate B axis 120 degrees at 900mm/min
        
        print("\nPress 'c' to test feed at current position")
        print("Then enter belt number (0-{}) that successfully fed".format(self.num_belts-1))
        print("(Counting 0,1,2,etc from left to right)")
        
        while True:
            key = input().lower()
            if key == 'c':
                # Calculate theta in degrees (converting from radians)
                theta = (6/self.radius) * (180/3.14159)
                
                # Back up by theta degrees while key is held
                cal.send_gcode_command(f"G1 B-{theta} F1200")
                
                # Move forward again by theta degrees
                cal.send_gcode_command(f"G1 B{theta} F600")
                
                print("\nPress 'c' to test feed again, or enter belt number (0-{}) that fed".format(self.num_belts-1))
            
            else:
                try:
                    belt_num = int(key)
                    if 0 <= belt_num < self.num_belts:
                        self.belt = belt_num
                        return belt_num
                    else:
                        print(f"Invalid belt number. Must be between 0 and {self.num_belts-1}")
                except ValueError:
                    if key != 'c':
                        print("Invalid input. Press 'c' to test or enter belt number")
        
        
    def feed(self, belt):
        """
        Feed material at the specified belt position.
        
        Args:
            belt (int): Belt number to feed from (0-N from left), or relative move (1,2) if not homed
            
        Raises:
            ValueError: If belt number is invalid
        """
        if not isinstance(belt, int):
            raise TypeError("belt must be an integer")
            
        cal = calibration.CalibrateToolheads()
        
        # Calculate rotation needed based on current and target belt positions
        if self.belt is None:
            # Not homed - treat belt as relative move indicator
            if belt == 1:
                rotation = -120
                print("Performing relative move: -120°")
            elif belt == 2:
                rotation = -240
                print("Performing relative move: -240°")
            elif 0 <= belt < self.num_belts:
                # Valid belt number, but not homed - feed at current position
                print("Warning: Feeder has not been homed - feeding at current position")
                rotation = 0
            else:
                raise ValueError(f"belt must be between 0 and {self.num_belts-1}, or 1/2 for relative moves when not homed")
            
            # Execute rotation if needed
            if rotation != 0:
                cal.send_gcode_command(f"G1 B{rotation} F900")
        else:
            # Homed - normal belt-based movement
            if not 0 <= belt < self.num_belts:
                raise ValueError(f"belt must be between 0 and {self.num_belts-1}")
                
            belt_diff = abs(belt - self.belt)
            
            # Determine rotation needed (-120° per position)
            if belt_diff == 1:
                rotation = -120
            elif belt_diff == 2:
                rotation = -240
            elif belt_diff > 2:
                # For differences >2, take shortest path
                rotation = -120 * min(belt_diff, self.num_belts - belt_diff)
                
            if belt_diff != 0:
                # Execute rotation
                cal.send_gcode_command(f"G1 B{rotation} F900")
                self.belt = belt
            
        # Calculate feed distance in degrees
        theta = (6/self.radius) * (180/3.14159)
        
        # Rock back then push forward to feed
        cal.send_gcode_command(f"G1 B-{theta} F1200")  # Back up quickly
        cal.send_gcode_command(f"G1 B{theta} F600")    # Feed forward slowly

        cal.close()


if __name__ == "__main__":
    import sys
    feeder = Feeder()

    if len(sys.argv) == 1:
        # No arguments - feed from current position
        feeder.feed(feeder.belt if feeder.belt is not None else 0)
    elif len(sys.argv) == 2:
        if sys.argv[1].lower() == "home":
            feeder.home()
            # After homing, ask user which belt they want to feed
            print(f"\nHoming complete. Which belt would you like to feed? (0-{feeder.num_belts-1})")
            while True:
                try:
                    belt_input = input("Enter belt number: ")
                    belt_num = int(belt_input)
                    if 0 <= belt_num < feeder.num_belts:
                        feeder.feed(belt_num)
                        break
                    else:
                        print(f"Invalid belt number. Must be between 0 and {feeder.num_belts-1}")
                except ValueError:
                    print("Invalid input. Please enter a valid belt number.")
        else:
            try:
                belt_num = int(sys.argv[1])
                feeder.feed(belt_num)
            except ValueError:
                print("Error: Argument must be an integer belt number or 'home'")
                sys.exit(1)
    else:
        print("Usage: python feed.py [belt_number|home]")
        sys.exit(1)
