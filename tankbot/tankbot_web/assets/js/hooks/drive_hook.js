/**
 * Keyboard + mouse controls for driving, arm, and grabber.
 *
 * Drive:   WASD / arrow keys (held = continuous)
 * Arm:     R (up) / F (down)
 * Grabber: T (close) / G (open)
 * Stop:    Space
 *
 * Buttons support press-and-hold via mousedown/mouseup + touchstart/touchend.
 * Servo commands repeat at REPEAT_MS while held.
 */

const SPEED = 2000
const TURN = 1500
const REPEAT_MS = 120

// Drive keys — summed while multiple are held
const DRIVE_KEYS = {
  w:          { left: SPEED, right: SPEED },
  arrowup:    { left: SPEED, right: SPEED },
  s:          { left: -SPEED, right: -SPEED },
  arrowdown:  { left: -SPEED, right: -SPEED },
  a:          { left: -TURN, right: TURN },
  arrowleft:  { left: -TURN, right: TURN },
  d:          { left: TURN, right: -TURN },
  arrowright: { left: TURN, right: -TURN },
}

// Servo keys — repeated while held
const SERVO_KEYS = {
  r: { event: "arm",     dir: "up" },
  f: { event: "arm",     dir: "down" },
  t: { event: "grabber", dir: "close" },
  g: { event: "grabber", dir: "open" },
}

export const DriveControls = {
  mounted() {
    this.drivePressed = new Set()
    this.servoIntervals = {}  // key → intervalId
    this.buttonInterval = null

    // --- Keyboard: drive ---
    this._onKeyDown = (e) => {
      const key = e.key.toLowerCase()

      if (key === " ") {
        e.preventDefault()
        this.pushEvent("stop", {})
        return
      }

      if (key in DRIVE_KEYS) {
        if (this.drivePressed.has(key)) return
        e.preventDefault()
        this.drivePressed.add(key)
        this._sendDrive()
        return
      }

      if (key in SERVO_KEYS) {
        if (this.servoIntervals[key]) return // already repeating
        e.preventDefault()
        const { event, dir } = SERVO_KEYS[key]
        this.pushEvent(event, { dir })
        this.servoIntervals[key] = setInterval(() => {
          this.pushEvent(event, { dir })
        }, REPEAT_MS)
        return
      }
    }

    this._onKeyUp = (e) => {
      const key = e.key.toLowerCase()

      if (key in DRIVE_KEYS) {
        e.preventDefault()
        this.drivePressed.delete(key)
        this._sendDrive()
        return
      }

      if (key in SERVO_KEYS && this.servoIntervals[key]) {
        clearInterval(this.servoIntervals[key])
        delete this.servoIntervals[key]
        return
      }
    }

    this._onBlur = () => {
      this.drivePressed.clear()
      this._clearAllServoIntervals()
      this._clearButtonInterval()
      this.pushEvent("stop", {})
    }

    window.addEventListener("keydown", this._onKeyDown)
    window.addEventListener("keyup", this._onKeyUp)
    window.addEventListener("blur", this._onBlur)

    // --- Button press-and-hold ---
    this._setupButtonHold()
  },

  destroyed() {
    window.removeEventListener("keydown", this._onKeyDown)
    window.removeEventListener("keyup", this._onKeyUp)
    window.removeEventListener("blur", this._onBlur)
    this._clearAllServoIntervals()
    this._clearButtonInterval()
  },

  _sendDrive() {
    if (this.drivePressed.size === 0) {
      this.pushEvent("stop", {})
      return
    }
    let left = 0, right = 0
    for (const key of this.drivePressed) {
      const m = DRIVE_KEYS[key]
      if (m) { left += m.left; right += m.right }
    }
    left = Math.max(-4095, Math.min(4095, left))
    right = Math.max(-4095, Math.min(4095, right))
    this.pushEvent("motor", { left, right })
  },

  _clearAllServoIntervals() {
    for (const key of Object.keys(this.servoIntervals)) {
      clearInterval(this.servoIntervals[key])
    }
    this.servoIntervals = {}
  },

  _clearButtonInterval() {
    if (this.buttonInterval) {
      clearInterval(this.buttonInterval)
      this.buttonInterval = null
    }
  },

  /** Make [data-hold-event] buttons repeat while pressed (mouse + touch). */
  _setupButtonHold() {
    const startHold = (e) => {
      const btn = e.target.closest("[data-hold-event]")
      if (!btn) return
      e.preventDefault()
      const event = btn.dataset.holdEvent
      const dir = btn.dataset.holdDir
      const payload = dir ? { dir } : {}
      // Also read phx-value-* for motor buttons
      const left = btn.getAttribute("phx-value-left")
      const right = btn.getAttribute("phx-value-right")
      if (left != null && right != null) {
        Object.assign(payload, { left: Number(left), right: Number(right) })
      }
      this.pushEvent(event, payload)
      this._clearButtonInterval()
      this.buttonInterval = setInterval(() => {
        this.pushEvent(event, payload)
      }, REPEAT_MS)
    }

    const stopHold = () => {
      this._clearButtonInterval()
    }

    this.el.addEventListener("mousedown", startHold)
    this.el.addEventListener("touchstart", startHold, { passive: false })
    window.addEventListener("mouseup", stopHold)
    window.addEventListener("touchend", stopHold)

    this._cleanupButtonHold = () => {
      this.el.removeEventListener("mousedown", startHold)
      this.el.removeEventListener("touchstart", startHold)
      window.removeEventListener("mouseup", stopHold)
      window.removeEventListener("touchend", stopHold)
    }
  },
}
