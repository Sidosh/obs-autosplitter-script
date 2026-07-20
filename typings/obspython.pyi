"""Type stub for OBS's built-in obspython module (editor only).

The real module is injected by OBS at runtime and does not exist as an
installable package. Functions used by this project are declared with
proper signatures; the module-level __getattr__ makes every other name
in the OBS API resolve as Any, so nothing gets flagged as missing.
"""

from typing import Any, Callable

LOG_ERROR: int
LOG_WARNING: int
LOG_INFO: int
LOG_DEBUG: int

def script_log(level: int, message: str) -> None: ...

def timer_add(callback: Callable[[], None], milliseconds: int) -> None: ...
def timer_remove(callback: Callable[[], None]) -> None: ...

def obs_frontend_start_virtualcam() -> None: ...
def obs_frontend_stop_virtualcam() -> None: ...
def obs_frontend_virtualcam_active() -> bool: ...

class obs_video_info:
    fps_num: int
    fps_den: int
    base_width: int
    base_height: int
    output_width: int
    output_height: int

def obs_get_video_info(ovi: obs_video_info) -> bool: ...

def obs_properties_create() -> Any: ...
def obs_properties_add_button(
    props: Any, name: str, text: str,
    callback: Callable[[Any, Any], bool],
) -> Any: ...

def __getattr__(name: str) -> Any: ...
