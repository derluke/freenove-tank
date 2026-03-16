/**
 * ExplorationMap hook — renders a top-down minimap on a <canvas> element.
 *
 * Receives map data from the server via push_event("map_update", data).
 * Data shape: { size, cell_cm, robot: [gx, gy, heading_deg], visited: [[gx,gy],...], obstacles: [[gx,gy],...] }
 */
export const ExplorationMap = {
  mounted() {
    this.canvas = this.el
    this.ctx = this.canvas.getContext("2d")
    this.mapData = null

    this.handleEvent("map_update", (data) => {
      this.mapData = data
      this.draw()
    })
  },

  updated() {
    if (this.mapData) this.draw()
  },

  draw() {
    const data = this.mapData
    if (!data) return

    const canvas = this.canvas
    const ctx = this.ctx
    const size = data.size || 100
    const w = canvas.width
    const h = canvas.height
    const cellW = w / size
    const cellH = h / size

    // Clear
    ctx.fillStyle = "#1a1a2e"
    ctx.fillRect(0, 0, w, h)

    // Draw grid lines (subtle)
    ctx.strokeStyle = "#2a2a4a"
    ctx.lineWidth = 0.5
    for (let i = 0; i <= size; i += 10) {
      ctx.beginPath()
      ctx.moveTo(i * cellW, 0)
      ctx.lineTo(i * cellW, h)
      ctx.stroke()
      ctx.beginPath()
      ctx.moveTo(0, i * cellH)
      ctx.lineTo(w, i * cellH)
      ctx.stroke()
    }

    // Draw visited cells (green, semi-transparent)
    ctx.fillStyle = "rgba(34, 197, 94, 0.4)"
    for (const [gx, gy] of data.visited || []) {
      // Flip y: grid y increases up, canvas y increases down
      ctx.fillRect(gx * cellW, (size - 1 - gy) * cellH, cellW, cellH)
    }

    // Draw obstacle cells (red)
    ctx.fillStyle = "rgba(239, 68, 68, 0.8)"
    for (const [gx, gy] of data.obstacles || []) {
      ctx.fillRect(gx * cellW, (size - 1 - gy) * cellH, cellW, cellH)
    }

    // Draw robot position and heading
    if (data.robot) {
      const [rx, ry, headingDeg] = data.robot
      const px = rx * cellW
      const py = (size - 1 - ry) * cellH + cellH / 2
      const headingRad = -headingDeg * Math.PI / 180  // negate for canvas coords

      ctx.save()
      ctx.translate(px, py)
      ctx.rotate(headingRad + Math.PI / 2)  // adjust: 0° = up in grid

      // Triangle pointing in heading direction
      const s = Math.max(cellW * 3, 6)
      ctx.fillStyle = "#60a5fa"
      ctx.beginPath()
      ctx.moveTo(0, -s)
      ctx.lineTo(-s * 0.6, s * 0.5)
      ctx.lineTo(s * 0.6, s * 0.5)
      ctx.closePath()
      ctx.fill()

      // White outline
      ctx.strokeStyle = "#fff"
      ctx.lineWidth = 1
      ctx.stroke()

      ctx.restore()
    }
  }
}
