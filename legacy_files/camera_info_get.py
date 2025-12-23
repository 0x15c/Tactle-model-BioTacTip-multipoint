import cv2 as cv

cap = cv.VideoCapture(0)

# List of properties to check
properties = {
    "CAP_PROP_FRAME_WIDTH": cv.CAP_PROP_FRAME_WIDTH,
    "CAP_PROP_FRAME_HEIGHT": cv.CAP_PROP_FRAME_HEIGHT,
    "CAP_PROP_FPS": cv.CAP_PROP_FPS,
    "CAP_PROP_BRIGHTNESS": cv.CAP_PROP_BRIGHTNESS,
    "CAP_PROP_CONTRAST": cv.CAP_PROP_CONTRAST,
    "CAP_PROP_SATURATION": cv.CAP_PROP_SATURATION,
    "CAP_PROP_HUE": cv.CAP_PROP_HUE,
    "CAP_PROP_GAIN": cv.CAP_PROP_GAIN,
    "CAP_PROP_EXPOSURE": cv.CAP_PROP_EXPOSURE,
    "CAP_PROP_FOCUS": cv.CAP_PROP_FOCUS,
    "CAP_PROP_AUTOFOCUS": cv.CAP_PROP_AUTOFOCUS,
}

# Print supported properties
for prop_name, prop_id in properties.items():
    value = cap.get(prop_id)
    if value != -1:  # -1 means the property is not supported
        print(f"{prop_name}: {value}")
    else:
        print(f"{prop_name}: Not supported")

cap.release()
'''
CAP_PROP_FRAME_WIDTH: 640.0
CAP_PROP_FRAME_HEIGHT: 480.0
CAP_PROP_FPS: 30.0
CAP_PROP_BRIGHTNESS: 0.0
CAP_PROP_CONTRAST: 32.0
CAP_PROP_SATURATION: 60.0
CAP_PROP_HUE: 0.0
CAP_PROP_GAIN: 0.0
CAP_PROP_EXPOSURE: -6.0
CAP_PROP_FOCUS: Not supported
CAP_PROP_AUTOFOCUS: Not supported
'''