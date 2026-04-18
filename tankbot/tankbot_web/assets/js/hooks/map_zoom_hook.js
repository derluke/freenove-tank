/**
 * MapZoom hook — mouse-wheel zoom, click-drag pan, and a live scale
 * bar overlay that stays fixed on screen regardless of zoom level.
 *
 * The server sends `data-ppm` (pixels-per-meter at 1x zoom) on the
 * container. The scale bar picks a round distance whose bar length is
 * ~25% of the container width at the current zoom, so the user can
 * always visually compare distances on the map.
 */

const NICE_DISTANCES = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0];

function pickScaleBar(ppmScreen, containerWidth) {
  // Target: bar should be ~25% of the container width in pixels.
  const targetPx = containerWidth * 0.25;
  const targetM = targetPx / ppmScreen;
  let best = NICE_DISTANCES[0];
  let bestDiff = Math.abs(best - targetM);
  for (const d of NICE_DISTANCES) {
    const diff = Math.abs(d - targetM);
    if (diff < bestDiff) {
      best = d;
      bestDiff = diff;
    }
  }
  return { meters: best, pixels: Math.round(best * ppmScreen) };
}

function formatDistance(m) {
  if (m < 0.1) return `${Math.round(m * 100)}cm`;
  if (m < 1.0) return `${m.toFixed(1)}m`;
  return `${m.toFixed(0)}m`;
}

export const MapZoom = {
  mounted() {
    this.scale = 1.0;
    this.tx = 0;
    this.ty = 0;
    this.dragging = false;
    this.dragStartX = 0;
    this.dragStartY = 0;
    this.dragStartTx = 0;
    this.dragStartTy = 0;

    const container = this.el;
    const img = container.querySelector("img");
    if (!img) return;
    this.img = img;

    this._fitImage();

    container.addEventListener("wheel", (e) => {
      e.preventDefault();
      const rect = container.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;

      const oldScale = this.scale;
      const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
      this.scale = Math.max(0.5, Math.min(20, this.scale * factor));

      this.tx = mx - (mx - this.tx) * (this.scale / oldScale);
      this.ty = my - (my - this.ty) * (this.scale / oldScale);
      this._apply();
    }, { passive: false });

    container.addEventListener("mousedown", (e) => {
      this.dragging = true;
      this.dragStartX = e.clientX;
      this.dragStartY = e.clientY;
      this.dragStartTx = this.tx;
      this.dragStartTy = this.ty;
    });

    window.addEventListener("mousemove", this._onMouseMove = (e) => {
      if (!this.dragging) return;
      this.tx = this.dragStartTx + (e.clientX - this.dragStartX);
      this.ty = this.dragStartTy + (e.clientY - this.dragStartY);
      this._apply();
    });

    window.addEventListener("mouseup", this._onMouseUp = () => {
      this.dragging = false;
    });

    container.addEventListener("dblclick", () => {
      this.scale = 1.0;
      this.tx = 0;
      this.ty = 0;
      this._fitImage();
    });
  },

  updated() {
    const img = this.el.querySelector("img");
    if (img) {
      this.img = img;
      this._apply();
    }
  },

  destroyed() {
    if (this._onMouseMove) window.removeEventListener("mousemove", this._onMouseMove);
    if (this._onMouseUp) window.removeEventListener("mouseup", this._onMouseUp);
  },

  _fitImage() {
    const container = this.el;
    const cw = container.clientWidth;
    const ch = container.clientHeight;
    const imgSize = Math.min(cw, ch);
    this.scale = imgSize / 480;
    this.tx = (cw - 480 * this.scale) / 2;
    this.ty = (ch - 480 * this.scale) / 2;
    this._apply();
  },

  _apply() {
    if (!this.img) return;
    this.img.style.width = "480px";
    this.img.style.height = "480px";
    this.img.style.transform = `translate(${this.tx}px, ${this.ty}px) scale(${this.scale})`;
    this._updateScaleBar();
  },

  _updateScaleBar() {
    const basePpm = parseFloat(this.el.dataset.ppm);
    if (!basePpm || basePpm <= 0) return;

    // Effective pixels-per-meter on screen = base ppm * current zoom scale.
    const ppmScreen = basePpm * this.scale;
    const cw = this.el.clientWidth;
    const bar = pickScaleBar(ppmScreen, cw);

    const label = this.el.querySelector("#map-scale-label");
    const line = this.el.querySelector("#map-scale-line");
    if (label) label.textContent = formatDistance(bar.meters);
    if (line) line.style.width = `${bar.pixels}px`;
  },
};
