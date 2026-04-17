/**
 * MapZoom hook — adds mouse-wheel zoom and click-drag pan to the
 * server-rendered occupancy map image.
 *
 * The image source is updated by LiveView via the normal DOM patch.
 * We override the img's CSS transform to apply zoom + translate,
 * keeping it responsive to new image payloads without losing the
 * user's current viewport.
 */
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

    // Fit image to container initially.
    this._fitImage();

    // Zoom on scroll.
    container.addEventListener("wheel", (e) => {
      e.preventDefault();
      const rect = container.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;

      const oldScale = this.scale;
      const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
      this.scale = Math.max(0.5, Math.min(10, this.scale * factor));

      // Zoom toward the mouse position.
      this.tx = mx - (mx - this.tx) * (this.scale / oldScale);
      this.ty = my - (my - this.ty) * (this.scale / oldScale);
      this._apply();
    }, { passive: false });

    // Pan on drag.
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

    // Double-click to reset.
    container.addEventListener("dblclick", () => {
      this.scale = 1.0;
      this.tx = 0;
      this.ty = 0;
      this._fitImage();
    });
  },

  updated() {
    // LiveView patched the DOM — the img src changed. Re-apply transform.
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
    // Center the image in the container at 1:1 scale.
    const container = this.el;
    const cw = container.clientWidth;
    const ch = container.clientHeight;
    // Image is square (480x480), fit it to the container.
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
  },
};
