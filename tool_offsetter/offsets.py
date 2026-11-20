"""
Turn measured tool length and diameter into CNC tool offsets (Fanuc, tool offset memory C:
separate geometry/wear values for H (length) and D (radius), as on the 0i Mate-MC offset screen).

  G10 L10 P<n> R<value>   H geometry (tool length)
  G10 L11 P<n> R<value>   H wear
  G10 L12 P<n> R<value>   D geometry (tool radius)
  G10 L13 P<n> R<value>   D wear

Length conventions
  "gauge"    : H = tool length measured from the spindle gauge line (positive). Work Z zero
               is then set from the gauge line (spindle nose) once for all tools.
  "touch_off": H = tool length - distance from the gauge line at machine Z home to the
               work Z zero, which gives the large negative values seen when tools are
               touched off on the part (e.g. -464.788).
"""


def length_offset(length_mm, convention="gauge", gauge_to_work_zero_mm=None):
    if convention == "gauge":
        return length_mm
    if convention == "touch_off":
        if gauge_to_work_zero_mm is None:
            raise ValueError("touch_off convention needs gauge_to_work_zero_mm")
        return length_mm - gauge_to_work_zero_mm
    raise ValueError(f"unknown convention: {convention}")


def fanuc_g10(tool, length_mm=None, diameter_mm=None, convention="gauge",
              gauge_to_work_zero_mm=None, wear=False):
    """G10 lines that load the offsets for one tool (offset number = tool number)."""
    if not isinstance(tool, int) or tool <= 0:
        raise ValueError("tool number must be a positive integer")
    lines = []
    if length_mm is not None:
        h = length_offset(length_mm, convention, gauge_to_work_zero_mm)
        lines.append(f"G10 L{11 if wear else 10} P{tool} R{h:.3f};")
    if diameter_mm is not None:
        lines.append(f"G10 L{13 if wear else 12} P{tool} R{diameter_mm / 2:.3f};")
    return lines


def fanuc_program(entries, program_number=9000, **kwargs):
    """Complete offset-loading program for several tools.
    entries: iterable of (tool_number, length_mm, diameter_mm)."""
    out = ["%", f"O{program_number:04d} (TOOL OFFSETS FROM VISION PRESETTER)", "G90;"]
    for tool, length, dia in entries:
        out += fanuc_g10(tool, length, dia, **kwargs)
    out += ["M30;", "%"]
    return "\n".join(out)
