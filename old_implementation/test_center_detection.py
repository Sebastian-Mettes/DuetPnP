#!/usr/bin/env python3
"""
Test script for center position detection in PnP Layer.
This script demonstrates how to find the center position (0,0) using circle detection.
"""

from PnP_Layer import PnPLayer
import json

def test_center_detection():
    """Test the center position detection functionality."""
    print("=== Center Position Detection Test ===")
    
    try:
        # Initialize PnP Layer (this will automatically run center detection)
        print("Initializing PnP Layer...")
        pnp = PnPLayer("placement_config.json", calibrate_tool=False)
        
        # Display the detected center position
        print(f"\nDetected center position:")
        print(f"X: {pnp.center['X']:.3f}")
        print(f"Y: {pnp.center['Y']:.3f}")
        print(f"Z: {pnp.center['Z']:.3f}")
        
        # Test placement coordinate calculation
        print(f"\nTesting placement coordinate calculation:")
        test_placement = {'x': 10, 'y': 20, 'z': 2.0, 'rotation': 0}
        print(f"Original placement: {test_placement}")
        
        # Simulate the calculation that happens in place_components
        final_x = test_placement['x'] + pnp.center['X']
        final_y = test_placement['y'] + pnp.center['Y']
        final_z = test_placement['z']
        
        print(f"Final coordinates (with center offset):")
        print(f"X: {final_x:.3f}")
        print(f"Y: {final_y:.3f}")
        print(f"Z: {final_z:.3f}")
        
        # Show the offset that was applied
        print(f"\nCenter offset applied:")
        print(f"X offset: {pnp.center['X']:.3f}")
        print(f"Y offset: {pnp.center['Y']:.3f}")
        
        print("\nCenter detection test completed successfully!")
        
    except Exception as e:
        print(f"Error during center detection test: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        try:
            pnp.cleanup()
        except:
            pass

def show_saved_center():
    """Show the saved center position from file."""
    try:
        with open('center_position.json', 'r') as f:
            center = json.load(f)
        print(f"\nSaved center position from file:")
        print(f"X: {center['X']:.3f}")
        print(f"Y: {center['Y']:.3f}")
        print(f"Z: {center['Z']:.3f}")
    except FileNotFoundError:
        print("No saved center position file found.")
    except Exception as e:
        print(f"Error reading saved center position: {str(e)}")

if __name__ == "__main__":
    print("Center Position Detection Test")
    print("==============================")
    
    # Show saved center position if it exists
    show_saved_center()
    
    # Run the test
    test_center_detection() 