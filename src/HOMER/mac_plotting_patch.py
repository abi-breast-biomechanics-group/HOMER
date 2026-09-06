"""
mac_plotting_patch.py – Work around macOS bus errors when closing a full-screen
PyVista window.

On macOS the VTK Cocoa render window is torn down (``vtkRenderWindow::Finalize``)
while the containing ``NSWindow`` may still be in a full-screen space.  AppKit
then messages the already-freed view during the exit-full-screen animation,
which surfaces as a ``Bus error: 10`` / ``EXC_BAD_ACCESS`` on interpreter exit.

The fix is simply to leave full screen *before* finalising, and to let the
window-server animation complete first.  Two kinds of "full screen" have to be
handled:

* VTK's own full screen (``Plotter.show(full_screen=True)`` or
  ``render_window.SetFullScreen(True)``) – undone through the VTK API.
* A native macOS full-screen space (the user hit the green button or
  ``ctrl-cmd-F``) – VTK knows nothing about this, so the ``NSWindow`` is asked
  to ``toggleFullScreen:`` through the Objective-C runtime via ``ctypes``.

Usage::

    from HOMER.mac_plotting_patch import apply_macos_fullscreen_close_patch
    apply_macos_fullscreen_close_patch()

The patch is idempotent and a no-op on non-Darwin platforms.
"""

import ctypes
import ctypes.util
import sys
import time

__all__ = ["apply_macos_fullscreen_close_patch", "exit_fullscreen"]

# NSWindowStyleMaskFullScreen – set while the window occupies a full-screen space.
_NS_FULLSCREEN_MASK = 1 << 14

# How long to keep pumping the event loop waiting for the un-full-screen
# animation to finish before giving up and closing anyway.
_ANIMATION_TIMEOUT = 2.0

_patched = False


# --------------------------------------------------------------------------- #
# Minimal Objective-C runtime bridge (avoids a hard pyobjc dependency)
# --------------------------------------------------------------------------- #
def _load_objc():
    """Return the libobjc handle, or ``None`` if it cannot be loaded."""
    try:
        path = ctypes.util.find_library("objc")
        if path is None:
            return None
        libobjc = ctypes.CDLL(path)
        libobjc.sel_registerName.restype = ctypes.c_void_p
        libobjc.sel_registerName.argtypes = [ctypes.c_char_p]
        return libobjc
    except OSError:
        return None


def _msg_send(libobjc, obj, selector, restype=ctypes.c_void_p, argtypes=(), args=()):
    """Send ``selector`` to the object at address ``obj``."""
    send = libobjc["objc_msgSend"]
    send.restype = restype
    send.argtypes = [ctypes.c_void_p, ctypes.c_void_p, *argtypes]
    sel = libobjc.sel_registerName(selector.encode())
    return send(ctypes.c_void_p(obj), ctypes.c_void_p(sel), *args)


def _vtk_pointer(handle):
    """Decode VTK's ``_00000001034e34b0_p_void`` pointer strings to an int."""
    if handle is None:
        return 0
    if isinstance(handle, int):
        return handle
    try:
        return int(str(handle).split("_")[1], 16)
    except (IndexError, ValueError):
        return 0


# --------------------------------------------------------------------------- #
# The actual work
# --------------------------------------------------------------------------- #
def _pump(plotter):
    """Give the window server a chance to run its animation."""
    iren = getattr(plotter, "iren", None)
    try:
        if iren is not None and iren.initialized:
            iren.process_events()
    except (RuntimeError, AttributeError):
        pass
    time.sleep(0.02)


def exit_fullscreen(plotter):
    """Take ``plotter``'s window out of full screen, if it is in one.

    Safe to call on any platform, on already-closed plotters and on off-screen
    plotters; it silently does nothing when there is nothing to do.
    """
    if sys.platform != "darwin":
        return

    try:
        render_window = plotter.render_window
    except (AttributeError, RuntimeError):
        return
    if render_window is None:
        return

    # 1. VTK-level full screen.
    try:
        if render_window.GetFullScreen():
            render_window.SetFullScreen(False)
            render_window.BordersOn()
            _pump(plotter)
    except (AttributeError, RuntimeError):
        pass

    # 2. Native macOS full-screen space.
    if render_window.GetClassName() != "vtkCocoaRenderWindow":
        return

    libobjc = _load_objc()
    if libobjc is None:
        return

    window = _vtk_pointer(render_window.GetRootWindow())
    if not window:
        return

    def style_mask():
        return _msg_send(libobjc, window, "styleMask", ctypes.c_ulong)

    try:
        if not style_mask() & _NS_FULLSCREEN_MASK:
            return
        _msg_send(
            libobjc,
            window,
            "toggleFullScreen:",
            None,
            argtypes=[ctypes.c_void_p],
            args=[None],
        )
        deadline = time.monotonic() + _ANIMATION_TIMEOUT
        while time.monotonic() < deadline and style_mask() & _NS_FULLSCREEN_MASK:
            _pump(plotter)
        # A couple of extra spins so the window has settled before Finalize().
        for _ in range(5):
            _pump(plotter)
    except Exception:  # never let the workaround break a close()
        pass


def apply_macos_fullscreen_close_patch():
    """Monkey-patch ``pyvista.Plotter.close`` to leave full screen first.

    Returns ``True`` if the patch was installed, ``False`` if it was skipped
    (non-macOS, PyVista unavailable, or already applied).
    """
    global _patched
    if _patched or sys.platform != "darwin":
        return False

    try:
        from pyvista.plotting.plotter import BasePlotter
    except ImportError:
        return False

    original_close = BasePlotter.close

    def close(self, *args, **kwargs):
        exit_fullscreen(self)
        return original_close(self, *args, **kwargs)

    close.__doc__ = original_close.__doc__
    close._homer_original_close = original_close
    BasePlotter.close = close
    _patched = True
    return True
