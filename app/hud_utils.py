import cv2
import numpy as np
import time
from src.config import Colors


def draw_corner_rect(img, pt1, pt2, color, thickness, length):
    """Draw stylish corner brackets around a rectangle."""
    x1, y1 = pt1
    x2, y2 = pt2

    # Top Left
    cv2.line(img, (x1, y1), (x1 + length, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + length), color, thickness)

    # Top Right
    cv2.line(img, (x2, y1), (x2 - length, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + length), color, thickness)

    # Bottom Left
    cv2.line(img, (x1, y2), (x1 + length, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - length), color, thickness)

    # Bottom Right
    cv2.line(img, (x2, y2), (x2 - length, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - length), color, thickness)


def draw_hud(img, labels, bboxes, confidences, fps=0, status="ACTIVE",
             inference_time=0, face_count=0):
    """Main function to draw a high-end Professional Sci-Fi HUD."""
    h, w = img.shape[:2]
    t = time.time()

    # ── 1. Background Grid (Sci-Fi Aesthetic) ─────────────────────────────
    grid_spacing = 60
    grid_img = np.zeros_like(img)
    for i in range(0, w, grid_spacing):
        cv2.line(grid_img, (i, 0), (i, h), (30, 30, 30), 1)
    for i in range(0, h, grid_spacing):
        cv2.line(grid_img, (0, i), (w, i), (30, 30, 30), 1)
    img = cv2.addWeighted(img, 1.0, grid_img, 0.2, 0)

    # ── 2. Overlay Panels (semi-transparent) ───────────────────────────────
    overlay = img.copy()
    # Top header bar
    cv2.rectangle(overlay, (0, 0), (w, 55), Colors.BG_DARK, -1)
    # Bottom status bar
    cv2.rectangle(overlay, (0, h - 30), (w, h), Colors.BG_DARK, -1)

    # Side Diagnostic Panel
    panel_w = 210
    panel_x = w - panel_w - 10
    panel_y1, panel_y2 = 65, 260
    cv2.rectangle(overlay, (panel_x, panel_y1), (w - 10, panel_y2), Colors.BG_PANEL, -1)
    cv2.rectangle(overlay, (panel_x, panel_y1), (w - 10, panel_y2), Colors.HUD_CYAN, 1)

    cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)

    # ── 3. Dynamic Scanline with trailing glow ─────────────────────────────
    scan_y = int((t * 250) % h)
    cv2.line(img, (0, scan_y), (w, scan_y), Colors.HUD_CYAN, 1)
    for i in range(1, 5):
        y_pos = scan_y - i * 2
        if y_pos >= 0:
            glow = img[y_pos].copy()
            img[y_pos] = cv2.addWeighted(
                img[y_pos], 1.0,
                np.full_like(img[y_pos], Colors.HUD_CYAN),
                0.15 / i, 0
            )

    # ── 4. Header Bar Content ──────────────────────────────────────────────
    # Title
    cv2.putText(img, "SYS // MASK_DETECTOR_G11", (25, 37),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, Colors.HUD_CYAN, 2, cv2.LINE_AA)

    # Pulsing REC light
    rec_alpha = (np.sin(t * 5) + 1) / 2
    rec_color = (0, 0, 255) if rec_alpha > 0.5 else (0, 0, 100)
    cv2.circle(img, (w - 185, 34), 6, rec_color, -1)
    cv2.putText(img, "LIVE FEED", (w - 170, 39),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, Colors.TEXT_WHITE, 1, cv2.LINE_AA)

    # ── 5. Diagnostics Panel ───────────────────────────────────────────────
    # Panel title + divider
    cv2.putText(img, "DIAGNOSTICS_CMD", (panel_x + 8, panel_y1 + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, Colors.HUD_CYAN, 1, cv2.LINE_AA)
    cv2.line(img, (panel_x + 5, panel_y1 + 30), (w - 15, panel_y1 + 30), Colors.HUD_CYAN, 1)

    # Metrics
    fps_color = Colors.MASK_GREEN if fps >= 25 else Colors.HUD_AMBER
    cv2.putText(img, f"AVG_FPS : {fps:.1f}", (panel_x + 10, panel_y1 + 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, fps_color, 1, cv2.LINE_AA)

    lat_ms = inference_time * 1000
    lat_color = Colors.MASK_GREEN if lat_ms < 30 else Colors.HUD_AMBER
    cv2.putText(img, f"INF_LAT : {lat_ms:.1f} ms", (panel_x + 10, panel_y1 + 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, lat_color, 1, cv2.LINE_AA)

    cv2.putText(img, f"TARGETS : {face_count}", (panel_x + 10, panel_y1 + 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, Colors.TEXT_WHITE, 1, cv2.LINE_AA)

    cv2.putText(img, "MODE    : REALTIME", (panel_x + 10, panel_y1 + 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, Colors.TEXT_GRAY, 1, cv2.LINE_AA)

    state_color = Colors.MASK_GREEN if status == "ACTIVE" else Colors.HUD_AMBER
    cv2.putText(img, f"STATE   : {status}", (panel_x + 10, panel_y1 + 155),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, state_color, 1, cv2.LINE_AA)

    # ── 6. Bottom Status Bar ───────────────────────────────────────────────
    cv2.putText(img, "[SPACE] PAUSE   [S] SCREENSHOT   [Q] QUIT",
                (15, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                Colors.TEXT_GRAY, 1, cv2.LINE_AA)

    # ── 7. Per-Detection Rendering ─────────────────────────────────────────
    for label, bbox, conf in zip(labels, bboxes, confidences):
        bx, by, bw, bh = bbox
        color = Colors.MASK_GREEN if label == "Mask" else Colors.NO_MASK_RED

        # Pulsing lock-on border (full rect, alpha-blended)
        pulse_alpha = (np.sin(t * 6) + 1) / 2 * 0.35 + 0.15   # 0.15 – 0.5
        pulse_overlay = img.copy()
        cv2.rectangle(pulse_overlay, (bx, by), (bx + bw, by + bh), color, 2)
        cv2.addWeighted(pulse_overlay, pulse_alpha, img, 1 - pulse_alpha, 0, img)

        # Corner brackets (drawn on top of pulse)
        draw_corner_rect(img, (bx, by), (bx + bw, by + bh), color, 2, 22)

        # Animated hex target ID
        target_id = hex(abs(hash(str(bx) + str(by))))[-6:].upper()

        # Label panel
        tag_w = 195
        tag_x2 = min(bx + tag_w, w - 2)
        tag_y1 = max(by - 48, 0)
        tag_y2 = max(by - 5, tag_y1 + 5)

        panel_overlay = img.copy()
        cv2.rectangle(panel_overlay, (bx, tag_y1), (tag_x2, tag_y2), Colors.BG_PANEL, -1)
        cv2.addWeighted(panel_overlay, 0.75, img, 0.25, 0, img)
        cv2.rectangle(img, (bx, tag_y1), (tag_x2, tag_y2), color, 1)

        # Connector line: panel corner → face box corner
        cv2.line(img, (bx + 5, tag_y2), (bx, by), color, 1, cv2.LINE_AA)

        # Label text + confidence %
        conf_pct = conf * 100
        cv2.putText(img, f"OBJ_{target_id} // {label.upper()}",
                    (bx + 6, tag_y1 + 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, Colors.TEXT_WHITE, 1, cv2.LINE_AA)
        cv2.putText(img, f"CONF: {conf_pct:.1f}%",
                    (bx + 6, tag_y1 + 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, color, 1, cv2.LINE_AA)

        # Confidence bar
        bar_max = tag_x2 - bx - 12
        bar_w = int(bar_max * conf)
        cv2.rectangle(img, (bx + 6, tag_y2 - 6), (bx + 6 + bar_max, tag_y2 - 3),
                      Colors.BG_DARK, -1)
        cv2.rectangle(img, (bx + 6, tag_y2 - 6), (bx + 6 + bar_w, tag_y2 - 3),
                      color, -1)

    return img
