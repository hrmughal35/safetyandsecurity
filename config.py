# Detection display threshold (balanced for CCTV footage)
DEFAULT_CONFIDENCE = 0.45

# Minimum score required before saving a violation image
DEFAULT_ALERT_CONFIDENCE = 0.48

# Scanned frames in a row required before saving (video/live)
DEFAULT_CONFIRM_FRAMES = 1

# Analyze every Nth frame in uploaded videos
DEFAULT_FRAME_STRIDE = 2

# Raw model query floor (do not set below this)
MODEL_SCAN_FLOOR = 0.25
