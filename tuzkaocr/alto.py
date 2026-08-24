import re
import unicodedata
from datetime import datetime, timezone
from xml.dom import minidom
from xml.etree import ElementTree as ET

_ID_SAFE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")
_ID_UNSAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def sanitize_xml_id(value: object, prefix: str = "id") -> str:
    candidate = f"{prefix}_{value}"
    if _ID_SAFE.fullmatch(candidate):
        return candidate
    folded = unicodedata.normalize("NFKD", candidate).encode("ascii", "ignore").decode("ascii")
    sanitized = _ID_UNSAFE.sub("_", folded).strip(".-") or f"{prefix}_value"
    if not sanitized[0].isalpha() and sanitized[0] != "_":
        sanitized = f"{prefix}_{sanitized}"
    return sanitized


def _clip_rect(x: object, y: object, width: object, height: object,
               page_w: int, page_h: int) -> tuple[int, int, int, int]:
    x0 = int(float(x))
    y0 = int(float(y))
    x1 = x0 + max(1, int(float(width)))
    y1 = y0 + max(1, int(float(height)))
    left = min(max(x0, 0), page_w - 1)
    top = min(max(y0, 0), page_h - 1)
    right = min(max(x1, left + 1), page_w)
    bottom = min(max(y1, top + 1), page_h)
    return left, top, right - left, bottom - top


def build_alto(page_id: str, img_h: int, img_w: int, blocks: list,
               software_name: str = "tuzkaocr",
               layout_name: str | None = None) -> str:
    img_h = int(img_h)
    img_w = int(img_w)
    if img_w < 1 or img_h < 1:
        raise ValueError(f"ALTO page dimensions must be positive, got {img_w}x{img_h}")

    alto = ET.Element("alto", {
        "xmlns": "http://www.loc.gov/standards/alto/ns-v4#",
        "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
        "xsi:schemaLocation": (
            "http://www.loc.gov/standards/alto/ns-v4# "
            "http://www.loc.gov/standards/alto/v4/alto-4-4.xsd"
        ),
    })

    desc = ET.SubElement(alto, "Description")
    mu = ET.SubElement(desc, "MeasurementUnit")
    mu.text = "pixel"

    now = datetime.now(timezone.utc).isoformat()

    if layout_name:
        ocr_layout = ET.SubElement(desc, "OCRProcessing", {"ID": "IdLayout"})
        layout_step = ET.SubElement(ocr_layout, "ocrProcessingStep")
        ET.SubElement(layout_step, "processingDateTime").text = now
        ET.SubElement(layout_step, "processingStepDescription").text = "layout"
        layout_sw = ET.SubElement(layout_step, "processingSoftware")
        ET.SubElement(layout_sw, "softwareCreator").text = "tuzkaocr"
        ET.SubElement(layout_sw, "softwareName").text = layout_name

    ocr_rec = ET.SubElement(desc, "OCRProcessing", {"ID": "IdRecognition"})
    step = ET.SubElement(ocr_rec, "ocrProcessingStep")
    ET.SubElement(step, "processingDateTime").text = now
    ET.SubElement(step, "processingStepDescription").text = "recognition"
    sw = ET.SubElement(step, "processingSoftware")
    ET.SubElement(sw, "softwareCreator").text = "tuzkaocr"
    ET.SubElement(sw, "softwareName").text = software_name

    used_roles = []
    for block in blocks:
        for line in block.get("lines") or []:
            r = line.get("role")
            if r and r != "body" and r not in used_roles:
                used_roles.append(r)
    role_tag_id = {r: f"ROLE_{r}" for r in used_roles}
    if used_roles:
        tags = ET.SubElement(alto, "Tags")
        for r in used_roles:
            ET.SubElement(tags, "StructureTag", {"ID": role_tag_id[r], "LABEL": r})
            
    layout = ET.SubElement(alto, "Layout")
    page = ET.SubElement(layout, "Page", {
        "ID": sanitize_xml_id(page_id, "page"),
        "WIDTH": str(img_w),
        "HEIGHT": str(img_h),
        "PHYSICAL_IMG_NR": "1",
    })
    ps = ET.SubElement(page, "PrintSpace", {
        "HPOS": "0", "VPOS": "0",
        "WIDTH": str(img_w), "HEIGHT": str(img_h),
    })

    for bi, block in enumerate(blocks):
        lines = block.get("lines")
        if not lines:
            continue
        line_rects = [
            _clip_rect(line["hpos"], line["vpos"], line["width"], line["height"], img_w, img_h)
            for line in lines
        ]
        bh = min(rect[0] for rect in line_rects)
        bv = min(rect[1] for rect in line_rects)
        br = max(rect[0] + rect[2] for rect in line_rects)
        bb = max(rect[1] + rect[3] for rect in line_rects)

        tb = ET.SubElement(ps, "TextBlock", {
            "ID": f"block_{bi}",
            "HPOS": str(bh), "VPOS": str(bv),
            "WIDTH": str(max(1, br - bh)), "HEIGHT": str(max(1, bb - bv)),
        })

        for li, (line, (lh, lv, lw, lht)) in enumerate(zip(lines, line_rects)):
            attrs = {
                "ID": f"line_{bi}_{li}",
                "HPOS": str(lh), "VPOS": str(lv),
                "WIDTH": str(lw), "HEIGHT": str(lht),
            }
            role = line.get("role")
            if role and role in role_tag_id:
                attrs["TAGREFS"] = role_tag_id[role]
            tl = ET.SubElement(tb, "TextLine", attrs)
            for wi, (word, wh, wv, ww, wht) in enumerate(line["words"]):
                wh, wv, ww, wht = _clip_rect(wh, wv, ww, wht, img_w, img_h)
                ET.SubElement(tl, "String", {
                    "ID": f"word_{bi}_{li}_{wi}",
                    "CONTENT": str(word),
                    "HPOS": str(wh), "VPOS": str(wv),
                    "WIDTH": str(ww), "HEIGHT": str(wht),
                })
                if wi < len(line["words"]) - 1:
                    ET.SubElement(tl, "SP")

    raw = ET.tostring(alto, encoding="unicode")
    return minidom.parseString(raw).toprettyxml(indent="  ")
