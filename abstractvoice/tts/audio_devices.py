"""Which speaker the audio comes out of — resolved LIVE, not from a stale cache.

PortAudio enumerates devices and resolves "the default output" inside
`Pa_Initialize()`, once. In a process that lives for hours — a desktop assistant —
that snapshot is taken at launch and never revised: a headset connected later is
invisible, and a default changed later is not seen. `sd.query_devices(None, "output")`
keeps naming the device that was default when the app started, so playback follows a
default that stopped existing hours ago. `Pa_Terminate()`/`Pa_Initialize()` refreshes
it (measured: ~4 ms) but invalidates EVERY open stream in the process, including an
open microphone — so it is a last resort, never a routine call.

macOS can be asked the live truth instead, for free: the CoreAudio HAL answers
"what is the default output right now" and gives every device a stable UID
(`AppleUSBAudioEngine:XREAL:XREAL One Pro:100000:6`) with no re-initialization and no
effect on open streams. This module asks CoreAudio what the answer is, then maps it
onto a PortAudio index by name, which is the only key PortAudio exposes.

UIDs are what a preference should store: an index is meaningless across a reboot and a
name is not unique, while a UID survives both and names the physical device.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import sys
from dataclasses import dataclass
from typing import Any, List, Optional

__all__ = [
    "OutputDevice",
    "list_output_devices",
    "system_default_output",
    "resolve_output_device",
    "describe_device_spec",
]


@dataclass(frozen=True)
class OutputDevice:
    """One device that can play audio, as both layers see it."""

    index: Optional[int]  # PortAudio index — None when PortAudio cannot see it (yet)
    name: str
    uid: str = ""  # CoreAudio UID; "" off macOS or when unavailable
    channels: int = 0
    samplerate: float = 0.0
    is_system_default: bool = False

    @property
    def key(self) -> str:
        """The stable identity to persist: UID when there is one, else the name."""
        return self.uid or self.name


# --------------------------------------------------------------------------- macOS


class _AudioObjectPropertyAddress(ctypes.Structure):
    _fields_ = [
        ("mSelector", ctypes.c_uint32),
        ("mScope", ctypes.c_uint32),
        ("mElement", ctypes.c_uint32),
    ]


def _fourcc(text: str) -> int:
    return int.from_bytes(text.encode("ascii"), "big")


_K_SYSTEM_OBJECT = 1
_SCOPE_GLOBAL = _fourcc("glob")
_SCOPE_OUTPUT = _fourcc("outp")
_ELEM_MAIN = 0
_PROP_DEFAULT_OUTPUT = _fourcc("dOut")
_PROP_DEVICES = _fourcc("dev#")
_PROP_NAME = _fourcc("lnam")
_PROP_UID = _fourcc("uid ")
_PROP_STREAM_CONFIG = _fourcc("slay")

_ca: Any = None
_cf: Any = None
_ca_loaded = False


def _coreaudio():
    """(CoreAudio, CoreFoundation) on macOS, or (None, None). Loaded once."""
    global _ca, _cf, _ca_loaded
    if _ca_loaded:
        return _ca, _cf
    _ca_loaded = True
    if sys.platform != "darwin":
        return None, None
    try:
        ca_path = ctypes.util.find_library("CoreAudio")
        cf_path = ctypes.util.find_library("CoreFoundation")
        if not ca_path or not cf_path:
            return None, None
        _ca = ctypes.cdll.LoadLibrary(ca_path)
        _cf = ctypes.cdll.LoadLibrary(cf_path)
        _cf.CFStringGetCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_long, ctypes.c_uint32]
        _cf.CFStringGetCString.restype = ctypes.c_bool
    except Exception:
        _ca, _cf = None, None
    return _ca, _cf


def _cfstring(ref) -> str:
    _, cf = _coreaudio()
    if cf is None or not ref:
        return ""
    buf = ctypes.create_string_buffer(512)
    try:
        if cf.CFStringGetCString(ref, buf, 512, 0x08000100):  # kCFStringEncodingUTF8
            return buf.value.decode("utf-8", "replace")
    except Exception:
        pass
    return ""


def _ca_uint32(obj_id: int, selector: int, scope: int = _SCOPE_GLOBAL) -> Optional[int]:
    ca, _ = _coreaudio()
    if ca is None:
        return None
    addr = _AudioObjectPropertyAddress(selector, scope, _ELEM_MAIN)
    size = ctypes.c_uint32(ctypes.sizeof(ctypes.c_uint32))
    value = ctypes.c_uint32()
    try:
        if ca.AudioObjectGetPropertyData(
            ctypes.c_uint32(obj_id), ctypes.byref(addr), 0, None, ctypes.byref(size), ctypes.byref(value)
        ) != 0:
            return None
    except Exception:
        return None
    return int(value.value)


def _ca_string(obj_id: int, selector: int) -> str:
    ca, cf = _coreaudio()
    if ca is None or cf is None:
        return ""
    addr = _AudioObjectPropertyAddress(selector, _SCOPE_GLOBAL, _ELEM_MAIN)
    size = ctypes.c_uint32(ctypes.sizeof(ctypes.c_void_p))
    ref = ctypes.c_void_p()
    try:
        if ca.AudioObjectGetPropertyData(
            ctypes.c_uint32(obj_id), ctypes.byref(addr), 0, None, ctypes.byref(size), ctypes.byref(ref)
        ) != 0:
            return ""
        try:
            return _cfstring(ref)
        finally:
            if ref:
                cf.CFRelease(ref)
    except Exception:
        return ""


def _ca_output_device_ids() -> List[int]:
    ca, _ = _coreaudio()
    if ca is None:
        return []
    addr = _AudioObjectPropertyAddress(_PROP_DEVICES, _SCOPE_GLOBAL, _ELEM_MAIN)
    size = ctypes.c_uint32()
    try:
        if ca.AudioObjectGetPropertyDataSize(
            ctypes.c_uint32(_K_SYSTEM_OBJECT), ctypes.byref(addr), 0, None, ctypes.byref(size)
        ) != 0:
            return []
        count = size.value // ctypes.sizeof(ctypes.c_uint32)
        if count <= 0:
            return []
        buf = (ctypes.c_uint32 * count)()
        if ca.AudioObjectGetPropertyData(
            ctypes.c_uint32(_K_SYSTEM_OBJECT), ctypes.byref(addr), 0, None, ctypes.byref(size), buf
        ) != 0:
            return []
    except Exception:
        return []

    out: List[int] = []
    for dev_id in buf:
        cfg = _AudioObjectPropertyAddress(_PROP_STREAM_CONFIG, _SCOPE_OUTPUT, _ELEM_MAIN)
        cfg_size = ctypes.c_uint32()
        try:
            if ca.AudioObjectGetPropertyDataSize(
                ctypes.c_uint32(dev_id), ctypes.byref(cfg), 0, None, ctypes.byref(cfg_size)
            ) != 0:
                continue
            # An AudioBufferList with no buffers is an input-only device.
            if cfg_size.value > ctypes.sizeof(ctypes.c_uint32) * 2:
                out.append(int(dev_id))
        except Exception:
            continue
    return out


# --------------------------------------------------------------------------- public


def system_default_output() -> Optional[OutputDevice]:
    """The output device the SYSTEM is sending audio to right now.

    On macOS this is the live CoreAudio answer, which costs a property read and does
    not disturb any open stream. Elsewhere it is PortAudio's answer, which is only as
    fresh as the last `Pa_Initialize()`.
    """
    ca, _ = _coreaudio()
    if ca is not None:
        dev_id = _ca_uint32(_K_SYSTEM_OBJECT, _PROP_DEFAULT_OUTPUT)
        if dev_id:
            name = _ca_string(dev_id, _PROP_NAME)
            if name:
                return OutputDevice(
                    index=_portaudio_index_for_name(name),
                    name=name,
                    uid=_ca_string(dev_id, _PROP_UID),
                    is_system_default=True,
                )
    try:
        info = _import_sounddevice().query_devices(None, "output")
        if isinstance(info, dict):
            index = info.get("index")
            return OutputDevice(
                index=int(index) if isinstance(index, int) else None,
                name=str(info.get("name") or "").strip(),
                channels=int(info.get("max_output_channels") or 0),
                samplerate=float(info.get("default_samplerate") or 0.0),
                is_system_default=True,
            )
    except Exception:
        pass
    return None


def _import_sounddevice():
    """The PortAudio binding, imported lazily.

    One seam, so a test can describe a machine (and so the player and this module
    always agree about which binding they are talking to).
    """
    import sounddevice as sd  # type: ignore

    return sd


def _portaudio_devices() -> List[dict]:
    try:
        return [dict(d) for d in _import_sounddevice().query_devices()]
    except Exception:
        return []


def _portaudio_index_for_name(name: str) -> Optional[int]:
    target = str(name or "").strip()
    if not target:
        return None
    for index, dev in enumerate(_portaudio_devices()):
        if int(dev.get("max_output_channels") or 0) <= 0:
            continue
        if str(dev.get("name") or "").strip() == target:
            return index
    return None


def list_output_devices() -> List[OutputDevice]:
    """Every device that can play audio, newest system truth first.

    PortAudio supplies the index (the only thing it can be opened by) and the channel
    and rate capabilities; CoreAudio supplies the UID and the live default flag. A
    device CoreAudio knows but PortAudio does not is still listed, with `index=None`:
    it was connected after PortAudio initialized, and saying so is better than hiding
    the very device the user is trying to pick.
    """
    default = system_default_output()
    default_name = default.name if default else ""

    devices: List[OutputDevice] = []
    seen_names: set[str] = set()
    for index, dev in enumerate(_portaudio_devices()):
        channels = int(dev.get("max_output_channels") or 0)
        if channels <= 0:
            continue
        name = str(dev.get("name") or "").strip()
        devices.append(
            OutputDevice(
                index=index,
                name=name,
                uid=_uid_for_name(name),
                channels=channels,
                samplerate=float(dev.get("default_samplerate") or 0.0),
                is_system_default=bool(default_name and name == default_name),
            )
        )
        seen_names.add(name)

    for dev_id in _ca_output_device_ids():
        name = _ca_string(dev_id, _PROP_NAME)
        if not name or name in seen_names:
            continue
        devices.append(
            OutputDevice(
                index=None,
                name=name,
                uid=_ca_string(dev_id, _PROP_UID),
                is_system_default=bool(default_name and name == default_name),
            )
        )
    return devices


_uid_cache: dict[str, str] = {}


def _uid_for_name(name: str) -> str:
    if not name:
        return ""
    if name in _uid_cache:
        return _uid_cache[name]
    uid = ""
    for dev_id in _ca_output_device_ids():
        if _ca_string(dev_id, _PROP_NAME) == name:
            uid = _ca_string(dev_id, _PROP_UID)
            break
    _uid_cache[name] = uid
    return uid


def resolve_output_device(spec: Any) -> Optional[OutputDevice]:
    """The device a spec names, as it exists RIGHT NOW.

    `spec` is None/"" (follow the system default), a UID, a device name, or a PortAudio
    index. Returns None when the spec names nothing currently present — which the
    caller must report rather than quietly playing somewhere else.
    """
    if spec is None or (isinstance(spec, str) and not spec.strip()):
        return system_default_output()

    devices = list_output_devices()
    if isinstance(spec, bool):  # bool is an int subclass; never an index
        return None
    if isinstance(spec, int):
        for device in devices:
            if device.index == int(spec):
                return device
        return None

    text = str(spec).strip()
    for device in devices:  # UID first: it is the stable identity
        if device.uid and device.uid == text:
            return device
    for device in devices:
        if device.name == text:
            return device
    return None


def describe_device_spec(spec: Any) -> str:
    """A human label for a stored preference, resolvable or not."""
    if spec is None or (isinstance(spec, str) and not spec.strip()):
        default = system_default_output()
        return f"System default ({default.name})" if default and default.name else "System default"
    device = resolve_output_device(spec)
    if device is not None:
        return device.name
    return f"{spec} (not connected)"
