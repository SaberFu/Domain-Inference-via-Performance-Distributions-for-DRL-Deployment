import os
import subprocess
import time
from typing import Any

from util.logger import Logger

def ensure_virtual_display(display: str = ":99") -> None:
    """Start Xvfb virtual display if no DISPLAY is set. Needed for headless Vulkan rendering.
    
    If DISPLAY is already set, uses it (assumes it's valid). Otherwise starts Xvfb on the
    specified display number.
    """
    if "DISPLAY" in os.environ:
        return
    
    try:
        process: subprocess.Popen[bytes] = subprocess.Popen(
            ["Xvfb", display, "-screen", "0", "1024x768x24"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        time.sleep(1)
        os.environ["DISPLAY"] = display
        Logger.print("Started virtual display on {}".format(display))
    except FileNotFoundError:
        Logger.print("WARNING: Xvfb not found. Install with: apt-get install xvfb")
        Logger.print("Headless camera rendering may not work without a virtual display.")
