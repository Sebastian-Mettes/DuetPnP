# Template Matching Test Utility

## Purpose

The `test_template_matching.py` script helps you diagnose and tune template matching parameters for your PnP system. It provides real-time visual feedback to ensure your camera setup and templates are optimal.

## Usage

```bash
python utils/test_template_matching.py <camera_num> <template_path>
```

### Examples

```bash
# Test upper camera (camera 2) with component template
python utils/test_template_matching.py 2 templates/resistor_0402_above.png

# Test lower camera (camera 0) with component template
python utils/test_template_matching.py 0 templates/resistor_0402_below.png
```

## Features

- **Live camera feed** with template matching overlay
- **Real-time match scores** and position information
- **Rotation angle detection** with configurable search range
- **Visual indicators**:
  - Green box: Good match (above threshold)
  - Orange box: Weak match (below threshold)
  - Red center dot: Detected component center
  - Blue crosshair: Image center
  - Yellow line: Offset from center
- **Template window** showing rotated template being matched
- **Frame freezing** for detailed inspection

## Keyboard Controls

| Key | Function |
|-----|----------|
| `SPACE` | Freeze/unfreeze current frame |
| `t` | Toggle template window on/off |
| `+/-` | Increase/decrease match threshold |
| `[/]` | Decrease/increase angle search step |
| `0-9` | Set exact angle (0=0°, 1=45°, 2=90°, etc.) |
| `a` | Enable full angular search |
| `s` | Save current frame and template to disk |
| `h` | Show help |
| `q/ESC` | Quit |

## Troubleshooting Guide

### Problem: "Could not parse detection result"

**Symptoms**: Detection works but centering fails with parse errors.

**Possible Causes**:
1. Template match score is below threshold (default 0.6)
2. Template doesn't match actual component appearance
3. Lighting conditions differ from when template was captured
4. Component is rotated beyond search range (-15° to +15°)

**Solutions**:
1. **Check match score**: Watch the live score display. If it's consistently below 0.6:
   - Press `-` to lower the threshold
   - If score is below 0.3, your template may not match

2. **Verify template quality**:
   - Capture a new template image from the actual camera view
   - Ensure template shows the component clearly
   - Template should be roughly the same size as component appears in camera

3. **Adjust lighting**:
   - Ensure LED ring lights are on
   - Check camera exposure settings
   - Reduce shadows and reflections

4. **Tune angle search**:
   - If component is at a known angle, use number keys (0-9) to test exact angles
   - Press `a` to enable full angular search
   - Use `[/]` to adjust angle search step size

### Problem: Low match scores (< 0.5)

**Solutions**:
1. Recapture template from the actual camera at the working distance
2. Ensure template is grayscale and clear
3. Verify component is in focus
4. Check that template size matches component size in camera view

### Problem: Multiple false detections

**Solutions**:
1. Increase match threshold with `+` key
2. Create a more distinctive template (include more unique features)
3. Improve lighting to increase contrast
4. Reduce background clutter in camera view

### Problem: Component not detected at all

**Solutions**:
1. Press `SPACE` to freeze frame and inspect
2. Verify component is actually in view
3. Check if component appears similar to template
4. Lower threshold with `-` key to see if any matches appear
5. Try different rotation angles with number keys

## Creating Good Templates

### Tips for Template Images:

1. **Capture from actual camera**: Use the same camera and distance as during operation
2. **Good lighting**: Consistent, even lighting with LED ring on
3. **Centered component**: Component should be centered in the template image
4. **Proper size**: Template should be cropped to just the component (minimal background)
5. **Clear features**: Include distinctive features of the component
6. **Correct orientation**: Template should match the expected component orientation
7. **Grayscale**: Templates should be saved as grayscale images

### Recommended Template Workflow:

1. Run this test script first to capture a frame:
   ```bash
   python utils/test_template_matching.py 2 templates/dummy.png
   ```

2. Press `s` to save the current frame

3. Open saved frame in image editor and crop to just the component

4. Save cropped image as your template (grayscale, PNG format)

5. Re-run the test script with your new template to verify match quality

## Expected Match Scores

- **0.8 - 1.0**: Excellent match, very reliable
- **0.6 - 0.8**: Good match, should work fine
- **0.4 - 0.6**: Weak match, may work but unreliable
- **< 0.4**: Poor match, template needs improvement

## Integration with PnP Workflow

Once you've verified your templates work well (scores > 0.6), they can be used in your `placement_config.json`:

```json
{
  "components": [
    {
      "type": "resistor_0402",
      "upper_template": "templates/resistor_0402_above.png",
      "lower_template": "templates/resistor_0402_below.png",
      ...
    }
  ]
}
```

The PnP workflow will use the same template matching algorithm you're testing here.
