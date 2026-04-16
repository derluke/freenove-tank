defmodule TankbotWebWeb.DashboardLive do
  use TankbotWebWeb, :live_view

  @impl true
  def mount(_params, _session, socket) do
    if connected?(socket) do
      Phoenix.PubSub.subscribe(TankbotWeb.PubSub, "robot:telemetry")
      Phoenix.PubSub.subscribe(TankbotWeb.PubSub, "robot:video")
    end

    {:ok,
     assign(socket,
       distance: 0.0,
       mode: 1,
       motor_left: 0,
       motor_right: 0,
       ir: 0,
       frame_data: nil,
       page_title: "Dashboard",
       # Vision engine state
       vision_active: false,
       vision_state: nil,
       vision_detections: [],
       vision_last_seen: nil,
       depth_left: 0.0,
       depth_center: 0.0,
       depth_right: 0.0,
       # SLAM state
       slam_tracking_quality: 0.0,
       slam_num_points: 0,
       slam_ply_epoch: nil,
       slam_ply_version: 0,
       autonomy_goal: nil,
       autonomy_behavior: nil,
       autonomy_phase: nil,
       planner_mode: nil,
       frontier_count: 0,
       coverage_ratio: 0.0,
       planner_reason: nil,
       selected_frontier: nil,
       status_detail: nil,
       lost_count: 0,
       scan_steps_remaining: 0,
       scan_round: 0,
       planner_target_heading_deg: nil,
       planner_target_cell: nil,
       force_live_ply: false,
       manual_ply_url: nil,
       manual_ply_label: nil,
       # Reactive engine state
       reactive_engine: false,
       reactive_stop_zone_blocked: false,
       reactive_stop_zone_count: 0,
       reactive_clearance_fwd: 0.0,
       reactive_clearance_left: 0.0,
       reactive_clearance_right: 0.0,
       reactive_obstacle_points: 0,
       reactive_motor_left: 0,
       reactive_motor_right: 0,
       reactive_depth_ms: 0.0,
       reactive_total_ms: 0.0,
       reactive_ultrasonic_m: nil,
       reactive_frame_count: 0,
       reactive_depth_image: nil,
       reactive_map_image: nil,
       reactive_pose_x: nil,
       reactive_pose_y: nil,
       reactive_pose_yaw: nil,
       reactive_pose_health: nil,
       reactive_map_cells: 0,
       # IMU (MPU-6050) latest reading
       imu_available: false,
       imu_ax: 0.0,
       imu_ay: 0.0,
       imu_az: 0.0,
       imu_gx: 0.0,
       imu_gy: 0.0,
       imu_gz: 0.0,
       imu_temp_c: nil
     )}
  end

  @impl true
  def handle_params(params, _uri, socket) do
    force_live_ply = params["live"] in ["1", "true", "yes"]

    ply_name =
      case params["ply"] do
        nil -> nil
        "" -> nil
        name -> name
      end

    socket =
      case ply_name do
        nil -> socket
        name -> push_event(socket, "splat_saved_ply", %{ply_url: "/assets/splat/saved/#{name}"})
      end

    {:noreply,
     assign(socket,
       force_live_ply: force_live_ply,
       manual_ply_url: if(ply_name, do: "/assets/splat/saved/#{ply_name}", else: nil),
       manual_ply_label: ply_name
     )}
  end

  @impl true
  def handle_info({:telemetry, %{"type" => "vision"} = data}, socket) do
    depth = data["depth"] || %{}
    slam = data["slam"] || %{}
    autonomy = data["autonomy"] || %{}

    socket =
      assign(socket,
        vision_active: true,
        vision_state: data["state"],
        vision_detections: data["detections"] || [],
        vision_last_seen: System.monotonic_time(:second),
        depth_left: depth["left"] || 0.0,
        depth_center: depth["center"] || 0.0,
        depth_right: depth["right"] || 0.0,
        slam_tracking_quality: slam["tracking_quality"] || 0.0,
        slam_num_points: slam["num_points"] || slam["num_gaussians"] || 0,
        slam_ply_epoch: slam["ply_epoch"] || socket.assigns.slam_ply_epoch,
        slam_ply_version: slam["ply_version"] || socket.assigns.slam_ply_version,
        autonomy_goal: autonomy["goal"] || socket.assigns.autonomy_goal,
        autonomy_behavior: autonomy["behavior"] || socket.assigns.autonomy_behavior,
        autonomy_phase: autonomy["phase"] || socket.assigns.autonomy_phase,
        planner_mode: autonomy["planner_mode"] || socket.assigns.planner_mode,
        frontier_count: autonomy["frontier_count"] || socket.assigns.frontier_count,
        coverage_ratio: autonomy["coverage_ratio"] || socket.assigns.coverage_ratio,
        planner_reason: autonomy["planner_reason"] || socket.assigns.planner_reason,
        selected_frontier: autonomy["selected_frontier"] || socket.assigns.selected_frontier,
        status_detail: autonomy["status_detail"] || socket.assigns.status_detail,
        lost_count: autonomy["lost_count"] || socket.assigns.lost_count,
        scan_steps_remaining: autonomy["scan_steps_remaining"] || socket.assigns.scan_steps_remaining,
        scan_round: autonomy["scan_round"] || socket.assigns.scan_round,
        planner_target_heading_deg: autonomy["planner_target_heading_deg"] || socket.assigns.planner_target_heading_deg,
        planner_target_cell: autonomy["planner_target_cell"] || socket.assigns.planner_target_cell,
        # Reactive engine fields (present when engine == "reactive")
        reactive_engine: autonomy["engine"] == "reactive",
        reactive_stop_zone_blocked: autonomy["stop_zone_blocked"] || false,
        reactive_stop_zone_count: autonomy["stop_zone_count"] || 0,
        reactive_clearance_fwd: autonomy["clearance_forward_m"] || 0.0,
        reactive_clearance_left: autonomy["clearance_left_m"] || 0.0,
        reactive_clearance_right: autonomy["clearance_right_m"] || 0.0,
        reactive_obstacle_points: autonomy["obstacle_points"] || 0,
        reactive_motor_left: autonomy["motor_left"] || 0,
        reactive_motor_right: autonomy["motor_right"] || 0,
        reactive_depth_ms: autonomy["depth_ms"] || 0.0,
        reactive_total_ms: autonomy["total_ms"] || 0.0,
        reactive_ultrasonic_m: autonomy["ultrasonic_m"],
        reactive_frame_count: autonomy["frame_count"] || 0,
        reactive_depth_image: autonomy["depth_image"] || socket.assigns.reactive_depth_image,
        reactive_map_image: autonomy["map_image"] || socket.assigns.reactive_map_image,
        reactive_pose_x: get_in(autonomy, ["pose", "x"]),
        reactive_pose_y: get_in(autonomy, ["pose", "y"]),
        reactive_pose_yaw: get_in(autonomy, ["pose", "yaw_deg"]),
        reactive_pose_health: get_in(autonomy, ["pose", "health"]),
        reactive_map_cells: autonomy["map_cells"] || 0
      )

    # Push camera pose to JS hook every frame (for robot marker)
    # Push PLY version only when it changes (for point cloud reload)
    socket =
      case slam["camera_pose"] do
        nil ->
          push_event(socket, "splat_update", %{
            ply_epoch: slam["ply_epoch"] || socket.assigns.slam_ply_epoch,
            ply_version: slam["ply_version"] || socket.assigns.slam_ply_version,
            camera_pose: nil,
            pose_valid: false,
            tracking_quality: slam["tracking_quality"] || 0.0,
            num_points: slam["num_points"] || slam["num_gaussians"] || 0
          })

        camera_pose ->
          push_event(socket, "splat_update", %{
            ply_epoch: slam["ply_epoch"] || socket.assigns.slam_ply_epoch,
            ply_version: slam["ply_version"] || socket.assigns.slam_ply_version,
            camera_pose: camera_pose,
            pose_valid: slam["pose_valid"] != false,
            tracking_quality: slam["tracking_quality"] || 0.0,
            num_points: slam["num_points"] || slam["num_gaussians"] || 0
          })
      end

    {:noreply, socket}
  end

  def handle_info({:telemetry, data}, socket) do
    # Check if vision engine has gone quiet (no update in 5s → inactive)
    vision_active =
      case socket.assigns.vision_last_seen do
        nil -> false
        ts -> System.monotonic_time(:second) - ts < 5
      end

    imu = data["imu"]

    socket =
      assign(socket,
        distance: data["distance"] || 0.0,
        mode: data["mode"] || 1,
        motor_left: get_in(data, ["motor", "left"]) || 0,
        motor_right: get_in(data, ["motor", "right"]) || 0,
        ir: data["ir"] || 0,
        vision_active: vision_active
      )

    socket =
      if imu do
        assign(socket,
          imu_available: true,
          imu_ax: imu["ax"] || 0.0,
          imu_ay: imu["ay"] || 0.0,
          imu_az: imu["az"] || 0.0,
          imu_gx: imu["gx"] || 0.0,
          imu_gy: imu["gy"] || 0.0,
          imu_gz: imu["gz"] || 0.0,
          imu_temp_c: imu["temp_c"]
        )
      else
        socket
      end

    {:noreply, socket}
  end

  def handle_info({:frame, jpeg_bytes}, socket) do
    b64 = Base.encode64(jpeg_bytes)
    {:noreply, assign(socket, frame_data: b64)}
  end

  @impl true
  def handle_event("motor", %{"left" => left, "right" => right}, socket) do
    TankbotWeb.RobotSocket.motor(left, right)
    {:noreply, socket}
  end

  def handle_event("stop", _params, socket) do
    TankbotWeb.RobotSocket.stop()
    {:noreply, socket}
  end

  def handle_event("servo", %{"channel" => ch, "angle" => angle}, socket) do
    TankbotWeb.RobotSocket.servo(ch, angle)
    {:noreply, socket}
  end

  def handle_event("led", %{"r" => r, "g" => g, "b" => b}, socket) do
    TankbotWeb.RobotSocket.led(r, g, b)
    {:noreply, socket}
  end

  def handle_event("led_off", _params, socket) do
    TankbotWeb.RobotSocket.led_off()
    {:noreply, socket}
  end

  def handle_event("mode", %{"mode" => mode}, socket) do
    TankbotWeb.RobotSocket.set_mode(mode)
    {:noreply, socket}
  end

  def handle_event("arm", %{"dir" => dir}, socket) do
    TankbotWeb.RobotSocket.arm(dir)
    {:noreply, socket}
  end

  def handle_event("grabber", %{"dir" => dir}, socket) do
    TankbotWeb.RobotSocket.grabber(dir)
    {:noreply, socket}
  end

  # --- Helpers for rendering ---

  defp vision_state_label("scanning"), do: "Scanning"
  defp vision_state_label("cruising"), do: "Cruising"
  defp vision_state_label("avoiding"), do: "Avoiding"
  defp vision_state_label("backing_up"), do: "Backing up"
  defp vision_state_label("stopped"), do: "Stopped"
  defp vision_state_label("forward"), do: "Forward"
  defp vision_state_label("turning"), do: "Turning"
  defp vision_state_label("reversing"), do: "Reversing"
  defp vision_state_label(_), do: "Unknown"

  defp vision_state_color("scanning"), do: "bg-blue-500"
  defp vision_state_color("cruising"), do: "bg-green-500"
  defp vision_state_color("avoiding"), do: "bg-yellow-500"
  defp vision_state_color("backing_up"), do: "bg-red-500"
  defp vision_state_color("stopped"), do: "bg-red-700"
  defp vision_state_color("forward"), do: "bg-green-500"
  defp vision_state_color("turning"), do: "bg-yellow-500"
  defp vision_state_color("reversing"), do: "bg-red-500"
  defp vision_state_color(_), do: "bg-gray-500"

  # d is distance in meters — lower = closer = more dangerous
  defp depth_bar_color(d) when d < 0.4, do: "bg-red-500"
  defp depth_bar_color(d) when d < 1.0, do: "bg-yellow-500"
  defp depth_bar_color(_), do: "bg-green-500"

  # Map meters to bar width (0m=100%, 3m+=0%)
  defp depth_bar_width(d), do: round(max(0, 100 - d / 3.0 * 100))

  defp distance_color(d) when d > 0 and d < 20, do: "text-red-400"
  defp distance_color(d) when d > 0 and d < 50, do: "text-yellow-400"
  defp distance_color(_), do: "text-green-400"

  defp panel_status_label(assigns) do
    cond do
      assigns.vision_active -> vision_state_label(assigns.vision_state)
      not is_nil(assigns.manual_ply_url) -> "Saved PLY"
      assigns.force_live_ply -> "Live PLY"
      true -> "Unknown"
    end
  end

  @impl true
  def render(assigns) do
    ~H"""
    <div class="min-h-screen bg-gray-900 text-white p-6">
      <div class="flex items-center gap-4 mb-6">
        <h1 class="text-3xl font-bold">TankBot Dashboard</h1>
        <%= if @vision_active do %>
          <span class={"inline-flex items-center gap-2 px-3 py-1 rounded-full text-sm font-semibold #{vision_state_color(@vision_state)}"}>
            <span class="relative flex h-2 w-2">
              <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-white opacity-75"></span>
              <span class="relative inline-flex rounded-full h-2 w-2 bg-white"></span>
            </span>
            <%= if @reactive_engine, do: "REACTIVE", else: "VISION" %>: <%= vision_state_label(@vision_state) %>
          </span>
        <% end %>
      </div>

      <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <%!-- Video feed + depth map --%>
        <div class="bg-gray-800 rounded-lg p-4">
          <%= if @reactive_engine and @reactive_depth_image do %>
            <%!-- Camera + Depth side by side (large) --%>
            <div class="grid grid-cols-2 gap-2">
              <div>
                <h2 class="text-sm font-semibold mb-1 text-gray-400">Camera</h2>
                <%= if @frame_data do %>
                  <img src={"data:image/jpeg;base64,#{@frame_data}"} class="w-full rounded" />
                <% else %>
                  <div class="w-full h-40 bg-gray-700 rounded flex items-center justify-center text-gray-400 text-sm">
                    Waiting for video...
                  </div>
                <% end %>
              </div>
              <div>
                <h2 class="text-sm font-semibold mb-1 text-gray-400">Depth</h2>
                <img src={"data:image/jpeg;base64,#{@reactive_depth_image}"} class="w-full rounded" />
              </div>
            </div>
            <%!-- Map below, full width --%>
            <%= if @reactive_map_image do %>
              <div class="mt-2">
                <h2 class="text-sm font-semibold mb-1 text-gray-400">Map</h2>
                <img src={"data:image/jpeg;base64,#{@reactive_map_image}"} class="w-full rounded max-h-64 mx-auto object-contain" />
              </div>
            <% end %>
          <% else %>
            <h2 class="text-xl font-semibold mb-3">Camera</h2>
            <div class="relative">
              <%= if @frame_data do %>
                <img src={"data:image/jpeg;base64,#{@frame_data}"} class="w-full rounded" />
              <% else %>
                <div class="w-full h-64 bg-gray-700 rounded flex items-center justify-center text-gray-400">
                  Waiting for video...
                </div>
              <% end %>
              <%!-- YOLO bounding boxes --%>
              <%= if @vision_active and @frame_data do %>
                <%= for det <- @vision_detections do %>
                  <div
                    class="absolute border-2 border-yellow-400 rounded-sm pointer-events-none"
                    style={"left:#{det["x1"]}%;top:#{det["y1"]}%;width:#{det["x2"] - det["x1"]}%;height:#{det["y2"] - det["y1"]}%"}
                  >
                    <span class="absolute -top-5 left-0 bg-yellow-400 text-black text-[10px] font-bold px-1 rounded-sm whitespace-nowrap">
                      <%= det["class"] %> <%= round(det["confidence"] * 100) %>% · <%= :erlang.float_to_binary((det["depth"] || 0) / 1, [decimals: 2]) %>m
                    </span>
                  </div>
                <% end %>
              <% end %>
            </div>
          <% end %>
        </div>

        <%!-- Telemetry + Controls --%>
        <div class="space-y-4">
          <div class="bg-gray-800 rounded-lg p-4">
            <h2 class="text-xl font-semibold mb-3">Sensors</h2>
            <div class="grid grid-cols-2 gap-4 text-sm">
              <div>
                <span class="text-gray-400">Distance:</span>
                <span class={"text-2xl font-mono ml-2 #{distance_color(@distance)}"}>
                  <%= :erlang.float_to_binary(@distance / 1, [decimals: 1]) %> cm
                </span>
              </div>
              <div>
                <span class="text-gray-400">IR Sensors:</span>
                <span class="text-2xl font-mono ml-2"><%= Integer.to_string(@ir, 2) |> String.pad_leading(3, "0") %></span>
              </div>
              <div>
                <span class="text-gray-400">Mode:</span>
                <span class="text-lg ml-2">
                  <%= case @mode do
                    1 -> "Manual"
                    2 -> "Sonar"
                    3 -> "Infrared"
                    _ -> "Mode #{@mode}"
                  end %>
                </span>
              </div>
              <div>
                <span class="text-gray-400">Motors:</span>
                <span class="font-mono ml-2">L:<%= @motor_left %> R:<%= @motor_right %></span>
              </div>
            </div>
          </div>

          <%!-- IMU panel --%>
          <%= if @imu_available do %>
            <div class="bg-gray-800 rounded-lg p-4 border border-indigo-500/30">
              <h2 class="text-xl font-semibold mb-3 flex items-center gap-2">
                IMU
                <span class="text-xs px-2 py-0.5 rounded-full bg-indigo-700">MPU-6050</span>
                <%= if @imu_temp_c do %>
                  <span class="text-xs text-gray-400 ml-auto font-mono"><%= :erlang.float_to_binary(@imu_temp_c / 1, [decimals: 1]) %>&deg;C</span>
                <% end %>
              </h2>
              <div class="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <div class="text-gray-400 text-xs mb-1">Accel (g)</div>
                  <div class="grid grid-cols-3 gap-2 font-mono text-xs">
                    <div><span class="text-gray-500">X</span> <span class="text-blue-200"><%= :erlang.float_to_binary(@imu_ax / 1, [decimals: 2]) %></span></div>
                    <div><span class="text-gray-500">Y</span> <span class="text-blue-200"><%= :erlang.float_to_binary(@imu_ay / 1, [decimals: 2]) %></span></div>
                    <div><span class="text-gray-500">Z</span> <span class="text-blue-200"><%= :erlang.float_to_binary(@imu_az / 1, [decimals: 2]) %></span></div>
                  </div>
                </div>
                <div>
                  <div class="text-gray-400 text-xs mb-1">Gyro (&deg;/s)</div>
                  <div class="grid grid-cols-3 gap-2 font-mono text-xs">
                    <div><span class="text-gray-500">X</span> <span class="text-emerald-200"><%= :erlang.float_to_binary(@imu_gx / 1, [decimals: 1]) %></span></div>
                    <div><span class="text-gray-500">Y</span> <span class="text-emerald-200"><%= :erlang.float_to_binary(@imu_gy / 1, [decimals: 1]) %></span></div>
                    <div><span class="text-gray-500">Z</span> <span class={"font-bold #{if abs(@imu_gz) > 5, do: "text-emerald-300", else: "text-emerald-200"}"}><%= :erlang.float_to_binary(@imu_gz / 1, [decimals: 1]) %></span></div>
                  </div>
                </div>
              </div>
            </div>
          <% end %>

          <%!-- Reactive autonomy panel --%>
          <%= if @reactive_engine and @vision_active do %>
            <div class="bg-gray-800 rounded-lg p-4 border border-emerald-500/30">
              <h2 class="text-xl font-semibold mb-3 flex items-center gap-2">
                Reactive Autonomy
                <span class={"text-xs px-2 py-0.5 rounded-full #{vision_state_color(@vision_state)}"}>
                  <%= vision_state_label(@vision_state) %>
                </span>
                <%= if @reactive_stop_zone_blocked do %>
                  <span class="text-xs px-2 py-0.5 rounded-full bg-red-600 animate-pulse">BLOCKED</span>
                <% end %>
              </h2>
              <div class="text-sm text-gray-300 space-y-3">
                <%!-- Clearance bars (left / forward / right) --%>
                <div>
                  <span class="text-gray-400 text-xs">Clearance (meters)</span>
                  <div class="flex gap-1 mt-1 h-6">
                    <div class="flex-1 bg-gray-700 rounded-sm overflow-hidden relative">
                      <div class={"h-full #{depth_bar_color(@reactive_clearance_left)} transition-all duration-150"} style={"width:#{depth_bar_width(@reactive_clearance_left)}%"}></div>
                      <span class="absolute inset-0 flex items-center justify-center text-[10px] font-mono">L <%= :erlang.float_to_binary(@reactive_clearance_left / 1, [decimals: 2]) %>m</span>
                    </div>
                    <div class="flex-1 bg-gray-700 rounded-sm overflow-hidden relative">
                      <div class={"h-full #{depth_bar_color(@reactive_clearance_fwd)} transition-all duration-150"} style={"width:#{depth_bar_width(@reactive_clearance_fwd)}%"}></div>
                      <span class="absolute inset-0 flex items-center justify-center text-[10px] font-mono">Fwd <%= :erlang.float_to_binary(@reactive_clearance_fwd / 1, [decimals: 2]) %>m</span>
                    </div>
                    <div class="flex-1 bg-gray-700 rounded-sm overflow-hidden relative">
                      <div class={"h-full #{depth_bar_color(@reactive_clearance_right)} transition-all duration-150"} style={"width:#{depth_bar_width(@reactive_clearance_right)}%"}></div>
                      <span class="absolute inset-0 flex items-center justify-center text-[10px] font-mono">R <%= :erlang.float_to_binary(@reactive_clearance_right / 1, [decimals: 2]) %>m</span>
                    </div>
                  </div>
                </div>
                <%!-- Motor output --%>
                <div class="grid grid-cols-2 gap-4">
                  <div>
                    <span class="text-gray-400 text-xs">Motor Output</span>
                    <div class="flex gap-3 mt-1">
                      <span class={"font-mono text-lg #{if @reactive_motor_left > 0, do: "text-green-400", else: if(@reactive_motor_left < 0, do: "text-red-400", else: "text-gray-500")}"}>
                        L:<%= @reactive_motor_left %>
                      </span>
                      <span class={"font-mono text-lg #{if @reactive_motor_right > 0, do: "text-green-400", else: if(@reactive_motor_right < 0, do: "text-red-400", else: "text-gray-500")}"}>
                        R:<%= @reactive_motor_right %>
                      </span>
                    </div>
                  </div>
                  <div>
                    <span class="text-gray-400 text-xs">Stop Zone</span>
                    <div class="mt-1">
                      <span class={"font-mono #{if @reactive_stop_zone_blocked, do: "text-red-400", else: "text-green-400"}"}>
                        <%= if @reactive_stop_zone_blocked, do: "BLOCKED", else: "Clear" %>
                      </span>
                      <span class="text-gray-500 text-xs ml-2">(<%= @reactive_stop_zone_count %> cells)</span>
                    </div>
                  </div>
                </div>
                <%!-- Current intent --%>
                <div class="rounded bg-gray-900/70 border border-gray-700 px-3 py-2">
                  <div class="text-[11px] uppercase tracking-wide text-gray-500 mb-1">Current Intent</div>
                  <div class="text-sm text-blue-100"><%= @status_detail || "Waiting..." %></div>
                </div>
                <%!-- Pose / odometry row --%>
                <%= if @reactive_pose_x do %>
                  <div class="rounded bg-gray-900/70 border border-gray-700 px-3 py-2">
                    <div class="text-[11px] uppercase tracking-wide text-gray-500 mb-1">Scan-Match Odometry</div>
                    <div class="flex flex-wrap gap-4 text-sm">
                      <span class="font-mono text-blue-200">x:<%= @reactive_pose_x %> y:<%= @reactive_pose_y %></span>
                      <span class="font-mono text-blue-200">yaw:<%= @reactive_pose_yaw %>&deg;</span>
                      <span class={"text-xs px-1.5 py-0.5 rounded #{if @reactive_pose_health == "healthy", do: "bg-green-700 text-green-100", else: if(@reactive_pose_health == "degraded", do: "bg-yellow-700 text-yellow-100", else: "bg-red-700 text-red-100")}"}>
                        <%= @reactive_pose_health %>
                      </span>
                      <span class="text-gray-500 text-xs">map: <%= @reactive_map_cells %> cells</span>
                    </div>
                  </div>
                <% end %>
                <%!-- Stats row --%>
                <div class="flex flex-wrap gap-4 text-xs text-gray-500">
                  <span>Depth: <span class="font-mono"><%= :erlang.float_to_binary(@reactive_depth_ms / 1, [decimals: 0]) %>ms</span></span>
                  <span>Tick: <span class="font-mono"><%= :erlang.float_to_binary(@reactive_total_ms / 1, [decimals: 1]) %>ms</span></span>
                  <span>Points: <span class="font-mono"><%= @reactive_obstacle_points %></span></span>
                  <span>US: <span class="font-mono"><%= if @reactive_ultrasonic_m, do: "#{:erlang.float_to_binary(@reactive_ultrasonic_m / 1, [decimals: 2])}m", else: "\u2014" %></span></span>
                  <span>Frame: <span class="font-mono">#<%= @reactive_frame_count %></span></span>
                </div>
              </div>
            </div>
          <% end %>

          <%!-- SLAM vision status panel (shown when SLAM engine is active, not reactive) --%>
          <%= if @vision_active and not @reactive_engine or (not is_nil(@manual_ply_url)) or @force_live_ply do %>
            <div class="bg-gray-800 rounded-lg p-4 border border-blue-500/30">
              <h2 class="text-xl font-semibold mb-3 flex items-center gap-2">
                Vision Autonomy
                <span class={"text-xs px-2 py-0.5 rounded-full #{vision_state_color(@vision_state)}"}>
                  <%= panel_status_label(assigns) %>
                </span>
              </h2>
              <div class="text-sm text-gray-300 space-y-3">
                <%!-- Depth proximity meter (left / center / right) in meters --%>
                <div>
                  <span class="text-gray-400 text-xs">Nearest obstacle (meters)</span>
                  <div class="flex gap-1 mt-1 h-6">
                    <div class="flex-1 bg-gray-700 rounded-sm overflow-hidden relative" title={"Left: #{:erlang.float_to_binary(@depth_left / 1, [decimals: 2])}m"}>
                      <div class={"h-full #{depth_bar_color(@depth_left)} transition-all duration-200"} style={"width:#{depth_bar_width(@depth_left)}%"}></div>
                      <span class="absolute inset-0 flex items-center justify-center text-[10px] font-mono">L <%= :erlang.float_to_binary(@depth_left / 1, [decimals: 1]) %>m</span>
                    </div>
                    <div class="flex-1 bg-gray-700 rounded-sm overflow-hidden relative" title={"Center: #{:erlang.float_to_binary(@depth_center / 1, [decimals: 2])}m"}>
                      <div class={"h-full #{depth_bar_color(@depth_center)} transition-all duration-200"} style={"width:#{depth_bar_width(@depth_center)}%"}></div>
                      <span class="absolute inset-0 flex items-center justify-center text-[10px] font-mono">C <%= :erlang.float_to_binary(@depth_center / 1, [decimals: 1]) %>m</span>
                    </div>
                    <div class="flex-1 bg-gray-700 rounded-sm overflow-hidden relative" title={"Right: #{:erlang.float_to_binary(@depth_right / 1, [decimals: 2])}m"}>
                      <div class={"h-full #{depth_bar_color(@depth_right)} transition-all duration-200"} style={"width:#{depth_bar_width(@depth_right)}%"}></div>
                      <span class="absolute inset-0 flex items-center justify-center text-[10px] font-mono">R <%= :erlang.float_to_binary(@depth_right / 1, [decimals: 1]) %>m</span>
                    </div>
                  </div>
                </div>

                <div>
                  <span class="text-gray-400">Objects detected:</span>
                  <span class="ml-1 font-mono"><%= length(@vision_detections) %></span>
                </div>
                <%= if length(@vision_detections) > 0 do %>
                  <div class="flex flex-wrap gap-2">
                    <%= for det <- @vision_detections do %>
                      <span class="bg-gray-700 px-2 py-1 rounded text-xs">
                        <span class="text-yellow-300"><%= det["class"] %></span>
                        <span class="text-gray-400 ml-1"><%= round(det["confidence"] * 100) %>%</span>
                      </span>
                    <% end %>
                  </div>
                <% end %>
              </div>
              <%!-- Exported Point-Cloud Map --%>
              <div class="mt-4">
                <h3 class="text-sm font-semibold text-gray-400 mb-2">3D Map</h3>
                <div
                  id="splat-viewer"
                  phx-hook="SplatViewer"
                  phx-update="ignore"
                  data-ply-epoch={@slam_ply_epoch}
                  data-ply-version={@slam_ply_version}
                  data-ply-url={@manual_ply_url}
                  class="w-full rounded border border-gray-700"
                  style="height: 400px;"
                ></div>
                <div class="flex gap-4 mt-2 text-xs text-gray-500">
                  <span>Points: <span class="font-mono"><%= @slam_num_points %></span></span>
                  <span>Tracking: <span class="font-mono"><%= if @slam_tracking_quality > 0, do: "#{round(@slam_tracking_quality * 100)}%", else: "\u2014" %></span></span>
                  <span>PLY: <span class="font-mono">v<%= @slam_ply_version %></span></span>
                  <span :if={@manual_ply_label}>Saved: <span class="font-mono"><%= @manual_ply_label %></span></span>
                </div>
                <div class="flex gap-4 mt-2 text-xs text-gray-500">
                  <span>Goal: <span class="font-mono"><%= @autonomy_goal || "\u2014" %></span></span>
                  <span>Behavior: <span class="font-mono"><%= @autonomy_behavior || "\u2014" %></span></span>
                  <span>Phase: <span class="font-mono"><%= @autonomy_phase || "\u2014" %></span></span>
                </div>
                <div class="flex gap-4 mt-2 text-xs text-gray-500">
                  <span>Planner: <span class="font-mono"><%= @planner_mode || "\u2014" %></span></span>
                  <span>Frontiers: <span class="font-mono"><%= @frontier_count %></span></span>
                  <span>Coverage: <span class="font-mono"><%= round(@coverage_ratio * 100) %>%</span></span>
                  <span>Reason: <span class="font-mono"><%= @planner_reason || "\u2014" %></span></span>
                </div>
                <div class="mt-3 rounded bg-gray-900/70 border border-gray-700 px-3 py-2">
                  <div class="text-[11px] uppercase tracking-wide text-gray-500 mb-1">Current Intent</div>
                  <div class="text-sm text-blue-100"><%= @status_detail || "Waiting for autonomy state…" %></div>
                  <div class="flex flex-wrap gap-4 mt-2 text-xs text-gray-400">
                    <span>Lost Frames: <span class="font-mono text-gray-200"><%= @lost_count %></span></span>
                    <span>Scan: <span class="font-mono text-gray-200">r<%= @scan_round %> / <%= @scan_steps_remaining %> left</span></span>
                    <span>Heading: <span class="font-mono text-gray-200"><%= if @planner_target_heading_deg != nil, do: "#{@planner_target_heading_deg}°", else: "\u2014" %></span></span>
                    <span>Target Cell: <span class="font-mono text-gray-200"><%= if @planner_target_cell, do: inspect(@planner_target_cell), else: "\u2014" %></span></span>
                  </div>
                </div>
              </div>
            </div>
          <% end %>

          <%!-- Controls --%>
          <div class="bg-gray-800 rounded-lg p-4">
            <h2 class="text-xl font-semibold mb-3">Drive</h2>
            <div class="grid grid-cols-3 gap-2 max-w-xs mx-auto select-none" id="drive-controls" phx-hook="DriveControls">
              <div></div>
              <button phx-click="motor" phx-value-left="2000" phx-value-right="2000"
                      data-hold-event="motor" class="bg-blue-600 hover:bg-blue-500 active:bg-blue-400 rounded p-3 text-center font-bold">
                <span class="block">&#9650;</span><span class="text-xs opacity-60">W</span>
              </button>
              <div></div>
              <button phx-click="motor" phx-value-left="-1500" phx-value-right="1500"
                      data-hold-event="motor" class="bg-blue-600 hover:bg-blue-500 active:bg-blue-400 rounded p-3 text-center font-bold">
                <span class="block">&#9668;</span><span class="text-xs opacity-60">A</span>
              </button>
              <button phx-click="stop"
                      class="bg-red-600 hover:bg-red-500 active:bg-red-400 rounded p-3 text-center font-bold">
                <span class="block">&#9632;</span><span class="text-xs opacity-60">Space</span>
              </button>
              <button phx-click="motor" phx-value-left="1500" phx-value-right="-1500"
                      data-hold-event="motor" class="bg-blue-600 hover:bg-blue-500 active:bg-blue-400 rounded p-3 text-center font-bold">
                <span class="block">&#9658;</span><span class="text-xs opacity-60">D</span>
              </button>
              <div></div>
              <button phx-click="motor" phx-value-left="-2000" phx-value-right="-2000"
                      data-hold-event="motor" class="bg-blue-600 hover:bg-blue-500 active:bg-blue-400 rounded p-3 text-center font-bold">
                <span class="block">&#9660;</span><span class="text-xs opacity-60">S</span>
              </button>
              <div></div>

              <%!-- Arm & Grabber inline --%>
              <div class="col-span-3 grid grid-cols-4 gap-2 mt-3 pt-3 border-t border-gray-700">
                <button phx-click="arm" phx-value-dir="up"
                        data-hold-event="arm" data-hold-dir="up"
                        class="bg-teal-600 hover:bg-teal-500 active:bg-teal-400 rounded p-3 font-bold text-center">
                  <span class="block">&#9650;</span><span class="text-xs opacity-60">R</span>
                </button>
                <button phx-click="arm" phx-value-dir="down"
                        data-hold-event="arm" data-hold-dir="down"
                        class="bg-teal-600 hover:bg-teal-500 active:bg-teal-400 rounded p-3 font-bold text-center">
                  <span class="block">&#9660;</span><span class="text-xs opacity-60">F</span>
                </button>
                <button phx-click="grabber" phx-value-dir="close"
                        data-hold-event="grabber" data-hold-dir="close"
                        class="bg-amber-600 hover:bg-amber-500 active:bg-amber-400 rounded p-3 font-bold text-center">
                  <span class="block">&#8847;&#8848;</span><span class="text-xs opacity-60">T</span>
                </button>
                <button phx-click="grabber" phx-value-dir="open"
                        data-hold-event="grabber" data-hold-dir="open"
                        class="bg-amber-600 hover:bg-amber-500 active:bg-amber-400 rounded p-3 font-bold text-center">
                  <span class="block">&#8848;&#8847;</span><span class="text-xs opacity-60">G</span>
                </button>
              </div>
              <p class="col-span-3 text-center text-gray-500 text-xs mt-1">
                Drive: WASD &middot; Arm: R/F &middot; Grab: T/G &middot; Stop: Space
              </p>
            </div>
          </div>

          <%!-- Mode selector --%>
          <div class="bg-gray-800 rounded-lg p-4">
            <h2 class="text-xl font-semibold mb-3">Mode</h2>
            <div class="flex gap-2">
              <button phx-click="mode" phx-value-mode="0" class="bg-green-700 hover:bg-green-600 px-4 py-2 rounded">Manual</button>
              <button phx-click="mode" phx-value-mode="1" class="bg-yellow-700 hover:bg-yellow-600 px-4 py-2 rounded">Sonar</button>
              <button phx-click="mode" phx-value-mode="2" class="bg-purple-700 hover:bg-purple-600 px-4 py-2 rounded">Line Follow</button>
            </div>
          </div>

          <%!-- Navigation --%>
          <div class="bg-gray-800 rounded-lg p-4">
            <.link navigate={~p"/blocks"} class="bg-orange-600 hover:bg-orange-500 px-6 py-3 rounded-lg text-lg font-bold inline-block">
              Block Programming
            </.link>
          </div>
        </div>
      </div>
    </div>
    """
  end
end
