/**
 * WASD + arrow key drive controls.
 *
 * Sends "motor" and "stop" events to the LiveView.
 * Keys are velocity-mapped so pressing multiple keys works intuitively.
 */

const SPEED = 2000
const TURN = 1500

const KEY_MAP = {
  w: { left: SPEED, right: SPEED },
  arrowup: { left: SPEED, right: SPEED },
  s: { left: -SPEED, right: -SPEED },
  arrowdown: { left: -SPEED, right: -SPEED },
  a: { left: -TURN, right: TURN },
  arrowleft: { left: -TURN, right: TURN },
  d: { left: TURN, right: -TURN },
  arrowright: { left: TURN, right: -TURN },
}

export const DriveControls = {
  mounted() {
    this.pressed = new Set()

    this._onKeyDown = (e) => {
      const key = e.key.toLowerCase()
      if (!(key in KEY_MAP)) return
      if (this.pressed.has(key)) return // held down, don't spam
      e.preventDefault()
      this.pressed.add(key)
      this._sendCurrent()
    }

    this._onKeyUp = (e) => {
      const key = e.key.toLowerCase()
      if (!(key in KEY_MAP)) return
      e.preventDefault()
      this.pressed.delete(key)
      this._sendCurrent()
    }

    // Also stop on window blur (user alt-tabs away)
    this._onBlur = () => {
      this.pressed.clear()
      this.pushEvent("stop", {})
    }

    window.addEventListener("keydown", this._onKeyDown)
    window.addEventListener("keyup", this._onKeyUp)
    window.addEventListener("blur", this._onBlur)
  },

  destroyed() {
    window.removeEventListener("keydown", this._onKeyDown)
    window.removeEventListener("keyup", this._onKeyUp)
    window.removeEventListener("blur", this._onBlur)
  },

  _sendCurrent() {
    if (this.pressed.size === 0) {
      this.pushEvent("stop", {})
      return
    }

    // Sum all pressed keys for combined movement (e.g. W+A = forward-left)
    let left = 0, right = 0
    for (const key of this.pressed) {
      const m = KEY_MAP[key]
      if (m) {
        left += m.left
        right += m.right
      }
    }

    // Clamp to valid range
    left = Math.max(-4095, Math.min(4095, left))
    right = Math.max(-4095, Math.min(4095, right))

    this.pushEvent("motor", { left, right })
  },
}
