/**
 * SplatViewer hook — renders the exported 3D point-cloud map.
 *
 * Loads the PLY exported by the autonomy SLAM adapter and displays
 * colored points using Three.js. Receives updates from the server
 * via push_event("splat_update", data).
 *
 * Data shape:
 *   { ply_version: int, camera_pose: [16 floats] | null, pose_valid: bool, tracking_quality: float, num_points: int }
 */
import * as THREE from "three"
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js"
import { PLYLoader } from "three/examples/jsm/loaders/PLYLoader.js"

export const SplatViewer = {
  mounted() {
    this.container = this.el
    this.currentEpoch = this.el.dataset.plyEpoch || null
    this.currentVersion = 0
    const urlParams = new URLSearchParams(window.location.search)
    const queryPly = urlParams.get("ply")
    this.liveMode = ["1", "true", "yes"].includes((urlParams.get("live") || "").toLowerCase())
    this.manualPlyUrl = queryPly ? `/assets/splat/saved/${queryPly}` : (this.el.dataset.plyUrl || null)
    this.renderer = null
    this.scene = null
    this.camera = null
    this.controls = null
    this.pointCloud = null
    this.robotMarker = null
    this.animFrameId = null
    this.loading = false
    this.livePollTimer = null
    this.hasInitialView = false
    this.plyLoader = new PLYLoader()
    // MASt3R-SLAM exports standard PLY with (x,y,z,red,green,blue)
    // PLYLoader reads these natively — no custom mapping needed

    this._initScene()
    this._initRobotMarker()
    this._showWaiting()
    this._animate()

    if (this.manualPlyUrl) {
      this._loadPLY(this.manualPlyUrl)
    } else if (this.liveMode) {
      this._startLivePolling()
    } else {
      const initialVersion = Number(this.el.dataset.plyVersion || 0)
      if (initialVersion > 0) {
        this.currentVersion = initialVersion
        this._loadPLY(this._plyUrl(this.currentEpoch, initialVersion))
      }
    }

    this.handleEvent("splat_update", (data) => {
      this._hideWaiting()

      if (this.manualPlyUrl) {
        if (data.pose_valid === false || !data.camera_pose) {
          this.robotMarker.visible = false
        } else if (data.camera_pose.length === 16) {
          this._updateRobotMarker(data.camera_pose)
        }
        return
      }

      if (data.ply_epoch && data.ply_epoch !== this.currentEpoch) {
        this.currentEpoch = data.ply_epoch
        this.currentVersion = 0
      }

      if (data.ply_version && data.ply_version > this.currentVersion && !this.loading) {
        this.currentVersion = data.ply_version
        this._loadPLY(this._plyUrl(this.currentEpoch, data.ply_version))
      }

      if (data.pose_valid === false || !data.camera_pose) {
        this.robotMarker.visible = false
      } else if (data.camera_pose.length === 16) {
        this._updateRobotMarker(data.camera_pose)
      }
    })

    this.handleEvent("splat_saved_ply", (data) => {
      if (!data?.ply_url) return
      this.manualPlyUrl = data.ply_url
      this.currentVersion = 0
      this._hideWaiting()
      this._loadPLY(this.manualPlyUrl)
    })
  },

  _plyUrl(epoch, version) {
    const params = new URLSearchParams()
    if (epoch) params.set("epoch", String(epoch))
    params.set("v", String(version))
    return `/assets/splat/scene.ply?${params.toString()}`
  },

  _livePlyUrl() {
    return `/assets/splat/scene.ply?t=${Date.now()}`
  },

  _startLivePolling() {
    this._loadPLY(this._livePlyUrl())
    this.livePollTimer = window.setInterval(() => {
      if (!this.loading) this._loadPLY(this._livePlyUrl())
    }, 3000)
  },

  _initScene() {
    const width = this.container.clientWidth
    const height = this.container.clientHeight || 400

    this.renderer = new THREE.WebGLRenderer({ antialias: true })
    this.renderer.setSize(width, height)
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    this.renderer.setClearColor(0x1a1a2e, 1)
    this.container.appendChild(this.renderer.domElement)

    this.scene = new THREE.Scene()

    this.camera = new THREE.PerspectiveCamera(60, width / height, 0.01, 200)
    this.camera.position.set(0, 3, 3)
    this.camera.lookAt(0, 0, 0)

    this.controls = new OrbitControls(this.camera, this.renderer.domElement)
    this.controls.enableDamping = true
    this.controls.dampingFactor = 0.1
    this.controls.minDistance = 0.1
    this.controls.maxDistance = 50
    this.controls.target.set(0, 0, 0)

    // Grid for spatial reference
    const grid = new THREE.GridHelper(10, 20, 0x333355, 0x222244)
    this.scene.add(grid)

    // Lighting for robot marker
    this.scene.add(new THREE.AmbientLight(0xffffff, 0.6))
    this.scene.add(new THREE.DirectionalLight(0xffffff, 0.4))

    this._resizeObserver = new ResizeObserver(() => {
      const w = this.container.clientWidth
      const h = this.container.clientHeight || 400
      this.camera.aspect = w / h
      this.camera.updateProjectionMatrix()
      this.renderer.setSize(w, h)
    })
    this._resizeObserver.observe(this.container)
  },

  _initRobotMarker() {
    const geometry = new THREE.ConeGeometry(0.08, 0.2, 8)
    geometry.rotateX(Math.PI / 2)
    const material = new THREE.MeshStandardMaterial({
      color: 0x60a5fa,
      emissive: 0x3080e0,
      emissiveIntensity: 0.5,
    })
    this.robotMarker = new THREE.Mesh(geometry, material)
    this.robotMarker.visible = false
    this.scene.add(this.robotMarker)

    this._trailPositions = []
    this._trailGeometry = new THREE.BufferGeometry()
    this._trailLine = new THREE.Line(
      this._trailGeometry,
      new THREE.LineBasicMaterial({ color: 0x60a5fa, opacity: 0.5, transparent: true })
    )
    this.scene.add(this._trailLine)
  },

  _showWaiting() {
    if (!this._waitingEl) {
      this._waitingEl = document.createElement("div")
      this._waitingEl.className = "absolute inset-0 flex items-center justify-center text-gray-500 text-sm"
      this._waitingEl.textContent = "Waiting for SLAM data..."
      this.container.style.position = "relative"
      this.container.appendChild(this._waitingEl)
    }
  },

  _hideWaiting() {
    if (this._waitingEl) {
      this._waitingEl.remove()
      this._waitingEl = null
    }
  },

  _loadPLY(url) {
    this.loading = true

    this.plyLoader.load(
      url,
      (geometry) => {
        // Remove old point cloud
        if (this.pointCloud) {
          this.scene.remove(this.pointCloud)
          this.pointCloud.geometry.dispose()
          this.pointCloud.material.dispose()
        }

        // MASt3R-SLAM PLY has standard (red, green, blue) — PLYLoader
        // reads these as geometry.attributes.color automatically
        const hasColor = !!geometry.attributes.color
        console.log("PLY attributes:", Object.keys(geometry.attributes), "hasColor:", hasColor)

        // Upstream MASt3R-SLAM exports use a different world convention than the
        // Tankbot live viewer. Match the existing viewer orientation in live mode.
        if (this.liveMode && geometry.attributes.position) {
          const pos = geometry.attributes.position
          for (let i = 0; i < pos.count; i++) {
            pos.setY(i, -pos.getY(i))
            pos.setZ(i, -pos.getZ(i))
          }
          pos.needsUpdate = true
          geometry.computeBoundingBox()
          geometry.computeBoundingSphere()
        }

        const material = new THREE.PointsMaterial({
          size: 0.008,
          vertexColors: hasColor,
          color: hasColor ? 0xffffff : 0x44aaff,
          sizeAttenuation: true,
        })

        this.pointCloud = new THREE.Points(geometry, material)
        this.pointCloud.frustumCulled = false
        this.scene.add(this.pointCloud)

        // Auto-center camera on the point cloud
        geometry.computeBoundingBox()
        const box = geometry.boundingBox
        if (box && !box.isEmpty()) {
          const center = box.getCenter(new THREE.Vector3())
          const size = box.getSize(new THREE.Vector3())
          const maxDim = Math.max(size.x, size.y, size.z)

          console.log(`PLY loaded: ${geometry.attributes.position.count} points, center:`, center, "size:", size)

          // Only auto-center once. Live upstream mode reloads the same scene.ply
          // repeatedly, so recentering every poll makes the camera jump.
          if (!this.hasInitialView) {
            this.controls.target.copy(center)
            this.camera.position.set(
              center.x + maxDim * 0.7,
              center.y + maxDim * 0.7,
              center.z + maxDim * 0.7,
            )
            this.camera.lookAt(center)
            this.hasInitialView = true
          }
        }

        this._hideWaiting()
        this.loading = false
      },
      undefined,
      (err) => {
        console.error("Failed to load PLY:", err)
        this.loading = false
      }
    )
  },

  _updateRobotMarker(poseFlat) {
    // poseFlat is T_WC (world-from-camera) 4x4 matrix in row-major order
    // Three.js uses column-major, so we transpose
    const m = new THREE.Matrix4()
    m.fromArray(poseFlat)
    m.transpose()

    // Extract position from T_WC (translation column)
    const pos = new THREE.Vector3()
    pos.setFromMatrixPosition(m)

    // Apply same Y/Z flip as PLY export (camera → Three.js convention)
    pos.y = -pos.y
    pos.z = -pos.z

    // Forward direction: camera looks along +Z in MASt3R convention
    const forward = new THREE.Vector3(0, 0, 1)
    forward.applyMatrix4(new THREE.Matrix4().extractRotation(m))
    forward.y = -forward.y
    forward.z = -forward.z

    this.robotMarker.position.copy(pos)
    this.robotMarker.lookAt(pos.x + forward.x, pos.y + forward.y, pos.z + forward.z)
    this.robotMarker.visible = true

    if (!this._markerLogCount) this._markerLogCount = 0
    if (this._markerLogCount++ % 60 === 0) {
      console.log("Robot pos:", pos.x.toFixed(3), pos.y.toFixed(3), pos.z.toFixed(3))
    }

    // Trail — throttled
    if (!this._trailCounter) this._trailCounter = 0
    this._trailCounter++
    if (this._trailCounter % 5 === 0) {
      this._trailPositions.push(pos.clone())
      if (this._trailPositions.length > 500) this._trailPositions.shift()

      const positions = new Float32Array(this._trailPositions.length * 3)
      for (let i = 0; i < this._trailPositions.length; i++) {
        positions[i * 3] = this._trailPositions[i].x
        positions[i * 3 + 1] = this._trailPositions[i].y
        positions[i * 3 + 2] = this._trailPositions[i].z
      }
      this._trailGeometry.setAttribute("position", new THREE.BufferAttribute(positions, 3))
      this._trailGeometry.computeBoundingSphere()
    }
  },

  _animate() {
    this.animFrameId = requestAnimationFrame(() => this._animate())
    if (this.controls) this.controls.update()
    if (this.renderer && this.scene && this.camera) {
      this.renderer.render(this.scene, this.camera)
    }
  },

  destroyed() {
    if (this.livePollTimer) window.clearInterval(this.livePollTimer)
    if (this.animFrameId) cancelAnimationFrame(this.animFrameId)
    if (this._resizeObserver) this._resizeObserver.disconnect()
    if (this.pointCloud) {
      this.scene.remove(this.pointCloud)
      this.pointCloud.geometry.dispose()
      this.pointCloud.material.dispose()
    }
    if (this.renderer) {
      this.renderer.dispose()
      if (this.renderer.domElement?.parentNode) {
        this.renderer.domElement.parentNode.removeChild(this.renderer.domElement)
      }
    }
    if (this.controls) this.controls.dispose()
  },
}
