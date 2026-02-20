#!/usr/bin/env python3
"""
DuetPnP Main Task Runner

This is the main entry point for running pick-and-place operations.
It loads configurations, initializes hardware, and executes the PnP workflow.

Usage:
    python run_pnp_task.py [placement_config.json] [--resume] [--offsets X Y] [--yolo MODEL]

Options:
    --resume              Resume from last checkpoint after error/interruption
    --offsets X Y         Apply X,Y offset (in mm) to all placements (e.g., --offsets -0.2 -0.15)
    --yolo MODEL          Use YOLO model for faster detection (e.g., --yolo YOLO/models/best.pt)
    --no-yolo             Disable YOLO even if model exists (use template matching)
    --no-training-capture Disable auto-saving detections for ML training (enabled by default)
"""

import sys
import os
import argparse
from machine_vision import load_camera_config
from machine_control import Printer, Feeder
from pnp_operations import ConfigManager, PnPWorkflow


def main():
    """Main entry point for PnP task execution."""
    # Parse command line arguments with argparse
    parser = argparse.ArgumentParser(
        description='DuetPnP Pick-and-Place System',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('config_file', nargs='?', default='placement_config.json',
                        help='Path to placement configuration file (default: placement_config.json)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from last checkpoint after error/interruption')
    parser.add_argument('--offsets', nargs=2, type=float, metavar=('X', 'Y'), default=None,
                        help='Apply X,Y offset in mm to all placements (e.g., --offsets -0.2 -0.15)')
    parser.add_argument('--yolo', type=str, metavar='MODEL', default=None,
                        help='Path to YOLO model for faster detection (e.g., YOLO/models/best.pt)')
    parser.add_argument('--no-yolo', action='store_true',
                        help='Disable YOLO detection (use template matching)')
    parser.add_argument('--no-training-capture', action='store_true',
                        help='Disable auto-saving of detections for ML training')

    args = parser.parse_args()

    config_file = args.config_file
    resume_mode = args.resume

    # Parse offsets if provided
    placement_offset = None
    if args.offsets:
        x_offset, y_offset = args.offsets
        placement_offset = {'x': x_offset, 'y': y_offset}
        print(f"Placement offset: X{x_offset:+.3f}, Y{y_offset:+.3f} mm")

    # Determine YOLO model path
    yolo_model_path = args.yolo
    use_yolo = not args.no_yolo
    
    # Check for default YOLO model if not specified
    if yolo_model_path is None and use_yolo:
        default_model = 'YOLO/models/best.pt'
        if os.path.exists(default_model):
            yolo_model_path = default_model
            print(f"Using default YOLO model: {yolo_model_path}")

    print("="*60)
    print("DuetPnP Pick-and-Place System")
    print("="*60)
    print(f"Configuration file: {config_file}")
    if resume_mode:
        print("Mode: RESUME from checkpoint")
    print()

    try:
        # 1. Load camera configurations
        print("Loading camera configurations...")
        cam0_config = load_camera_config(0)  # Lower/upward camera
        cam2_config = load_camera_config(2)  # Upper/downward camera
        camera_configs = {0: cam0_config, 2: cam2_config}
        print("✓ Camera configurations loaded\n")

        # 2. Load placement configuration
        print("Loading placement configuration...")
        config_manager = ConfigManager(config_file)
        print("✓ Placement configuration loaded\n")

        # 3. Initialize printer
        print("Connecting to printer...")
        printer = Printer(upward_camera_number=0, debug=False)
        print("✓ Printer connected\n")

        # 4. Initialize and home feeder
        print("Skipping initializing feeder...")
        feeder = Feeder(num_belts=3, radius=11.45)
        #print("Homing feeder (follow on-screen instructions)...")
        #feeder.home(printer)
        #print("✓ Feeder ready\n")

        # 5. Initialize PnP workflow
        print("Initializing PnP workflow...")
        enable_training_capture = not args.no_training_capture
        workflow = PnPWorkflow(
            printer=printer,
            feeder=feeder,
            config_manager=config_manager,
            camera_configs=camera_configs,
            placement_offset=placement_offset,
            yolo_model_path=yolo_model_path,
            use_yolo=use_yolo,
            enable_training_capture=enable_training_capture
        )
        print("✓ Workflow initialized\n")

        # 6. Check for existing checkpoint if not in resume mode
        if not resume_mode and os.path.exists('pnp_checkpoint.json'):
            import json
            with open('pnp_checkpoint.json', 'r') as f:
                checkpoint = json.load(f)
            if checkpoint.get('status') == 'paused':
                print("\n" + "!"*60)
                print("WARNING: Found existing checkpoint from previous run!")
                print("!"*60)
                response = input("\nDo you want to:\n  [R]esume from checkpoint\n  [S]tart fresh (discard checkpoint)\n> ").strip().upper()
                if response in ['R', 'RESUME']:
                    resume_mode = True
                    print("\n✓ Resuming from checkpoint...")
                else:
                    print("\n✓ Starting fresh, checkpoint will be overwritten...")

        # 7. Run the workflow
        print("="*60)
        success = workflow.run(resume=resume_mode)
        print("="*60)

        if success:
            print("\n✓ PnP task completed successfully!")
        else:
            print("\n⚠ PnP task ended with errors or was aborted")
            return 1

    except KeyboardInterrupt:
        print("\n\n" + "!"*60)
        print("Task interrupted by user (Ctrl+C)")
        print("!"*60)
        try:
            workflow.save_checkpoint()
            print("\n✓ Progress saved to checkpoint")
            print("  Run with --resume to continue from this point")
        except:
            print("\n✗ Could not save checkpoint")
        return 1

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        # Clean up
        try:
            printer.close()
        except:
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
