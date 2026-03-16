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
       page_title: "Dashboard"
     )}
  end

  @impl true
  def handle_info({:telemetry, data}, socket) do
    {:noreply,
     assign(socket,
       distance: data["distance"] || 0.0,
       mode: data["mode"] || 1,
       motor_left: get_in(data, ["motor", "left"]) || 0,
       motor_right: get_in(data, ["motor", "right"]) || 0,
       ir: data["ir"] || 0
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

  @impl true
  def render(assigns) do
    ~H"""
    <div class="min-h-screen bg-gray-900 text-white p-6">
      <h1 class="text-3xl font-bold mb-6">TankBot Dashboard</h1>

      <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <%!-- Video feed --%>
        <div class="bg-gray-800 rounded-lg p-4">
          <h2 class="text-xl font-semibold mb-3">Camera</h2>
          <%= if @frame_data do %>
            <img src={"data:image/jpeg;base64,#{@frame_data}"} class="w-full rounded" />
          <% else %>
            <div class="w-full h-64 bg-gray-700 rounded flex items-center justify-center text-gray-400">
              Waiting for video...
            </div>
          <% end %>
        </div>

        <%!-- Telemetry --%>
        <div class="space-y-4">
          <div class="bg-gray-800 rounded-lg p-4">
            <h2 class="text-xl font-semibold mb-3">Sensors</h2>
            <div class="grid grid-cols-2 gap-4 text-sm">
              <div>
                <span class="text-gray-400">Distance:</span>
                <span class="text-2xl font-mono ml-2"><%= :erlang.float_to_binary(@distance / 1, [decimals: 1]) %> cm</span>
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

          <%!-- Controls --%>
          <div class="bg-gray-800 rounded-lg p-4">
            <h2 class="text-xl font-semibold mb-3">Drive</h2>
            <div class="grid grid-cols-3 gap-2 max-w-xs mx-auto select-none" id="drive-controls" phx-hook="DriveControls">
              <div></div>
              <button phx-click="motor" phx-value-left="2000" phx-value-right="2000"
                      data-hold-event="motor" class="bg-blue-600 hover:bg-blue-500 active:bg-blue-400 rounded p-3 text-center font-bold">
                <span class="block">▲</span><span class="text-xs opacity-60">W</span>
              </button>
              <div></div>
              <button phx-click="motor" phx-value-left="-1500" phx-value-right="1500"
                      data-hold-event="motor" class="bg-blue-600 hover:bg-blue-500 active:bg-blue-400 rounded p-3 text-center font-bold">
                <span class="block">◄</span><span class="text-xs opacity-60">A</span>
              </button>
              <button phx-click="stop"
                      class="bg-red-600 hover:bg-red-500 active:bg-red-400 rounded p-3 text-center font-bold">
                <span class="block">■</span><span class="text-xs opacity-60">Space</span>
              </button>
              <button phx-click="motor" phx-value-left="1500" phx-value-right="-1500"
                      data-hold-event="motor" class="bg-blue-600 hover:bg-blue-500 active:bg-blue-400 rounded p-3 text-center font-bold">
                <span class="block">►</span><span class="text-xs opacity-60">D</span>
              </button>
              <div></div>
              <button phx-click="motor" phx-value-left="-2000" phx-value-right="-2000"
                      data-hold-event="motor" class="bg-blue-600 hover:bg-blue-500 active:bg-blue-400 rounded p-3 text-center font-bold">
                <span class="block">▼</span><span class="text-xs opacity-60">S</span>
              </button>
              <div></div>

              <%!-- Arm & Grabber inline --%>
              <div class="col-span-3 grid grid-cols-4 gap-2 mt-3 pt-3 border-t border-gray-700">
                <button phx-click="arm" phx-value-dir="up"
                        data-hold-event="arm" data-hold-dir="up"
                        class="bg-teal-600 hover:bg-teal-500 active:bg-teal-400 rounded p-3 font-bold text-center">
                  <span class="block">▲</span><span class="text-xs opacity-60">R</span>
                </button>
                <button phx-click="arm" phx-value-dir="down"
                        data-hold-event="arm" data-hold-dir="down"
                        class="bg-teal-600 hover:bg-teal-500 active:bg-teal-400 rounded p-3 font-bold text-center">
                  <span class="block">▼</span><span class="text-xs opacity-60">F</span>
                </button>
                <button phx-click="grabber" phx-value-dir="close"
                        data-hold-event="grabber" data-hold-dir="close"
                        class="bg-amber-600 hover:bg-amber-500 active:bg-amber-400 rounded p-3 font-bold text-center">
                  <span class="block">⊏⊐</span><span class="text-xs opacity-60">T</span>
                </button>
                <button phx-click="grabber" phx-value-dir="open"
                        data-hold-event="grabber" data-hold-dir="open"
                        class="bg-amber-600 hover:bg-amber-500 active:bg-amber-400 rounded p-3 font-bold text-center">
                  <span class="block">⊐⊏</span><span class="text-xs opacity-60">G</span>
                </button>
              </div>
              <p class="col-span-3 text-center text-gray-500 text-xs mt-1">
                Drive: WASD · Arm: R/F · Grab: T/G · Stop: Space
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
              🧩 Block Programming
            </.link>
          </div>
        </div>
      </div>
    </div>
    """
  end
end
