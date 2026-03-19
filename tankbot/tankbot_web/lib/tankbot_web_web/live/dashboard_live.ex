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
       slam_ply_version: 0,
       autonomy_goal: nil,
       autonomy_behavior: nil,
       autonomy_phase: nil
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
        slam_ply_version: slam["ply_version"] || socket.assigns.slam_ply_version,
        autonomy_goal: autonomy["goal"] || socket.assigns.autonomy_goal,
        autonomy_behavior: autonomy["behavior"] || socket.assigns.autonomy_behavior,
        autonomy_phase: autonomy["phase"] || socket.assigns.autonomy_phase
      )

    # Push camera pose to JS hook every frame (for robot marker)
    # Push PLY version only when it changes (for point cloud reload)
    socket =
      case slam["camera_pose"] do
        nil ->
          push_event(socket, "splat_update", %{
            ply_version: slam["ply_version"] || socket.assigns.slam_ply_version,
            camera_pose: nil,
            pose_valid: false,
            tracking_quality: slam["tracking_quality"] || 0.0,
            num_points: slam["num_points"] || slam["num_gaussians"] || 0
          })

        camera_pose ->
          push_event(socket, "splat_update", %{
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

    {:noreply,
     assign(socket,
       distance: data["distance"] || 0.0,
       mode: data["mode"] || 1,
       motor_left: get_in(data, ["motor", "left"]) || 0,
       motor_right: get_in(data, ["motor", "right"]) || 0,
       ir: data["ir"] || 0,
       vision_active: vision_active
     )}
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
  defp vision_state_label(_), do: "Unknown"

  defp vision_state_color("scanning"), do: "bg-blue-500"
  defp vision_state_color("cruising"), do: "bg-green-500"
  defp vision_state_color("avoiding"), do: "bg-yellow-500"
  defp vision_state_color("backing_up"), do: "bg-red-500"
  defp vision_state_color("stopped"), do: "bg-red-700"
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
            VISION: <%= vision_state_label(@vision_state) %>
          </span>
        <% end %>
      </div>

      <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <%!-- Video feed --%>
        <div class="bg-gray-800 rounded-lg p-4">
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

          <%!-- Vision status panel --%>
          <%= if @vision_active do %>
            <div class="bg-gray-800 rounded-lg p-4 border border-blue-500/30">
              <h2 class="text-xl font-semibold mb-3 flex items-center gap-2">
                Vision Autonomy
                <span class={"text-xs px-2 py-0.5 rounded-full #{vision_state_color(@vision_state)}"}><%= vision_state_label(@vision_state) %></span>
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
                  class="w-full rounded border border-gray-700"
                  style="height: 400px;"
                ></div>
                <div class="flex gap-4 mt-2 text-xs text-gray-500">
                  <span>Points: <span class="font-mono"><%= @slam_num_points %></span></span>
                  <span>Tracking: <span class="font-mono"><%= if @slam_tracking_quality > 0, do: "#{round(@slam_tracking_quality * 100)}%", else: "\u2014" %></span></span>
                  <span>PLY: <span class="font-mono">v<%= @slam_ply_version %></span></span>
                </div>
                <div class="flex gap-4 mt-2 text-xs text-gray-500">
                  <span>Goal: <span class="font-mono"><%= @autonomy_goal || "\u2014" %></span></span>
                  <span>Behavior: <span class="font-mono"><%= @autonomy_behavior || "\u2014" %></span></span>
                  <span>Phase: <span class="font-mono"><%= @autonomy_phase || "\u2014" %></span></span>
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
