"""
Generate training curve visual artifacts (ASCII and SVG) for Stage C baseline.
"""
import json
from pathlib import Path

def generate_visual_artifacts():
    exp_dir = Path("data/experiments/stage_c_baseline")
    curve_data_path = exp_dir / "loss_curve.json"
    if not curve_data_path.is_file():
        print(f"Error: {curve_data_path} not found")
        return

    with open(curve_data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    train_loss = data["train_loss"]
    val_evals = data["val_evaluations"]

    # 1. ASCII plot
    lines = [
        "========================================================================",
        "       CHAKRMICRO v0.1 — STAGE C BASELINE TRAINING & VALIDATION CURVE   ",
        "========================================================================",
        "",
        "Loss ^",
    ]

    rows = 11
    y_max = 8.5
    y_min = 3.5
    y_step = (y_max - y_min) / (rows - 1)

    cols = 50
    col_size = len(train_loss) // cols
    t_sampled = [sum(train_loss[i * col_size : (i + 1) * col_size]) / col_size for i in range(cols)]

    grid = [[" " for _ in range(cols)] for _ in range(rows)]

    # Plot train loss
    for c, val in enumerate(t_sampled):
        r = int(round((y_max - val) / y_step))
        if 0 <= r < rows:
            grid[r][c] = "-"

    # Plot val loss
    for v in val_evals:
        step = v["step"]
        val = v["val_loss"]
        c = min(cols - 1, int(round((step / 500.0) * (cols - 1))))
        r = int(round((y_max - val) / y_step))
        if 0 <= r < rows:
            grid[r][c] = "O"

    for i in range(rows):
        y_val = y_max - i * y_step
        row_str = "".join(grid[i])
        lines.append(f"{y_val:4.1f} | {row_str}")

    lines.append("     +" + "-" * cols)
    lines.append("Step:  0         100       200       300       400       500")
    lines.append("")
    lines.append("Legend: '-' = Train Loss (smoothed), 'O' = Validation Loss (periodic batches)")
    lines.append("")

    ascii_txt = "\n".join(lines)
    (exp_dir / "loss_curve.txt").write_text(ascii_txt, encoding="utf-8")

    # 2. SVG Vector plot
    width = 800
    height = 500
    pad_left = 70
    pad_right = 40
    pad_top = 50
    pad_bottom = 60

    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom

    def map_x(step):
        return pad_left + (step / 500.0) * plot_w

    def map_y(loss):
        return pad_top + ((y_max - loss) / (y_max - y_min)) * plot_h

    # SVG paths
    train_pts = []
    for step_idx, l in enumerate(train_loss, start=1):
        x = map_x(step_idx)
        y = map_y(l)
        train_pts.append(f"{x:.1f},{y:.1f}")
    train_path_d = "M " + " L ".join(train_pts)

    val_pts = []
    val_circles = []
    for v in val_evals:
        x = map_x(v["step"])
        y = map_y(v["val_loss"])
        val_pts.append(f"{x:.1f},{y:.1f}")
        val_circles.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="#f43f5e" stroke="#ffffff" stroke-width="1.5"/>')
    val_path_d = "M " + " L ".join(val_pts)

    # Grid lines and labels
    grid_svg = []
    for i in range(rows):
        y_val = y_max - i * y_step
        y_pos = map_y(y_val)
        grid_svg.append(f'<line x1="{pad_left}" y1="{y_pos:.1f}" x2="{width - pad_right}" y2="{y_pos:.1f}" stroke="#334155" stroke-dasharray="3,3" stroke-width="0.8"/>')
        grid_svg.append(f'<text x="{pad_left - 10}" y="{y_pos + 4:.1f}" fill="#94a3b8" font-size="12" text-anchor="end" font-family="monospace">{y_val:.1f}</text>')

    x_labels = [0, 100, 200, 300, 400, 500]
    for step in x_labels:
        x_pos = map_x(step)
        grid_svg.append(f'<line x1="{x_pos:.1f}" y1="{pad_top}" x2="{x_pos:.1f}" y2="{height - pad_bottom}" stroke="#334155" stroke-dasharray="3,3" stroke-width="0.8"/>')
        grid_svg.append(f'<text x="{x_pos:.1f}" y="{height - pad_bottom + 20}" fill="#94a3b8" font-size="12" text-anchor="middle" font-family="monospace">{step}</text>')

    svg_content = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" style="background-color: #0f172a;">
  <!-- Title -->
  <text x="{width/2}" y="30" fill="#f8fafc" font-size="16" font-weight="bold" text-anchor="middle" font-family="sans-serif">ChakrMicro v0.1 — Stage C Baseline Training Curve</text>
  
  <!-- Axis Grid -->
  {''.join(grid_svg)}
  
  <!-- Axes Lines -->
  <line x1="{pad_left}" y1="{pad_top}" x2="{pad_left}" y2="{height - pad_bottom}" stroke="#64748b" stroke-width="1.5"/>
  <line x1="{pad_left}" y1="{height - pad_bottom}" x2="{width - pad_right}" y2="{height - pad_bottom}" stroke="#64748b" stroke-width="1.5"/>
  
  <!-- Axis Titles -->
  <text x="{width/2}" y="{height - 15}" fill="#cbd5e1" font-size="13" text-anchor="middle" font-family="sans-serif">Optimizer Steps (Tokens = Steps × 1,024)</text>
  <text x="20" y="{height/2}" fill="#cbd5e1" font-size="13" text-anchor="middle" font-family="sans-serif" transform="rotate(-90 20 {height/2})">Cross-Entropy Loss</text>
  
  <!-- Train Loss Curve -->
  <path d="{train_path_d}" fill="none" stroke="#38bdf8" stroke-width="1.2" opacity="0.85"/>
  
  <!-- Val Loss Curve and Points -->
  <path d="{val_path_d}" fill="none" stroke="#f43f5e" stroke-width="2.0"/>
  {''.join(val_circles)}
  
  <!-- Legend -->
  <rect x="{width - pad_right - 210}" y="{pad_top + 10}" width="200" height="60" rx="4" fill="#1e293b" stroke="#334155" opacity="0.9"/>
  <line x1="{width - pad_right - 195}" y1="{pad_top + 30}" x2="{width - pad_right - 170}" y2="{pad_top + 30}" stroke="#38bdf8" stroke-width="2"/>
  <text x="{width - pad_right - 160}" y="{pad_top + 34}" fill="#e2e8f0" font-size="12" font-family="sans-serif">Train Loss (step)</text>
  <circle cx="{width - pad_right - 182}" cy="{pad_top + 50}" r="4" fill="#f43f5e"/>
  <text x="{width - pad_right - 160}" y="{pad_top + 54}" fill="#e2e8f0" font-size="12" font-family="sans-serif">Validation Loss (batch)</text>
</svg>"""

    (exp_dir / "loss_curve.svg").write_text(svg_content, encoding="utf-8")
    print(f"Generated {exp_dir / 'loss_curve.txt'} and {exp_dir / 'loss_curve.svg'}")

if __name__ == "__main__":
    generate_visual_artifacts()
