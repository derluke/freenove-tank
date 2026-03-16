defmodule TankbotWebWeb.BlocksLive do
  @moduledoc """
  Blockly-based visual programming interface.

  IMPORTANT: This LiveView avoids server-side assign changes that trigger DOM
  patches, because Blockly creates internal SVG elements with duplicate IDs
  that cause LiveView warnings on every patch. All dynamic UI (distance, run/stop
  button toggle, log) is handled via JS push_event + client-side DOM manipulation.
  """
  use TankbotWebWeb, :live_view

  @impl true
  def mount(_params, _session, socket) do
    if connected?(socket) do
      Phoenix.PubSub.subscribe(TankbotWeb.PubSub, "robot:telemetry")
    end

    {:ok, assign(socket, page_title: "Block Programming")}
  end

  @impl true
  def handle_info({:telemetry, data}, socket) do
    distance = data["distance"] || 0.0
    {:noreply, push_event(socket, "distance_update", %{distance: distance})}
  end

  @impl true
  # Robot commands from client-side Blockly execution
  def handle_event("motor", %{"left" => left, "right" => right}, socket) do
    TankbotWeb.RobotSocket.motor(left, right)
    {:noreply, socket}
  end

  def handle_event("stop", _params, socket) do
    TankbotWeb.RobotSocket.stop()
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

  def handle_event("servo", %{"channel" => ch, "angle" => angle}, socket) do
    TankbotWeb.RobotSocket.servo(ch, angle)
    {:noreply, socket}
  end

  def handle_event("stop_program", _params, socket) do
    TankbotWeb.RobotSocket.stop()
    {:noreply, push_event(socket, "stop_program", %{})}
  end

  # No-ops — these are handled client-side, but LiveView needs handlers
  def handle_event("program_started", _p, socket), do: {:noreply, socket}
  def handle_event("program_done", _p, socket), do: {:noreply, socket}
  def handle_event("log_entry", _p, socket), do: {:noreply, socket}
  def handle_event("run_program", _p, socket), do: {:noreply, socket}

  @impl true
  def render(assigns) do
    ~H"""
    <div class="min-h-screen bg-gray-900 text-white">
      <%!-- Header --%>
      <div class="bg-gray-800 p-4 flex items-center justify-between">
        <div class="flex items-center gap-4">
          <.link navigate={~p"/"} class="text-gray-400 hover:text-white">Back</.link>
          <h1 class="text-2xl font-bold">Block Programming</h1>
        </div>
        <div class="flex items-center gap-4">
          <span id="blocks-distance" class="text-sm text-gray-400">Distance: -- cm</span>
          <div class="flex items-center gap-2">
            <select id="program-select" class="bg-gray-700 text-white text-sm rounded px-3 py-2 border border-gray-600">
              <option value="">-- Programs --</option>
            </select>
            <button id="load-btn" class="px-3 py-2 rounded text-sm bg-gray-600 hover:bg-gray-500">Load</button>
            <button id="save-btn" class="px-3 py-2 rounded text-sm bg-blue-600 hover:bg-blue-500">Save</button>
            <button id="delete-btn" class="px-3 py-2 rounded text-sm bg-gray-600 hover:bg-gray-500">Del</button>
          </div>
          <div class="flex items-center gap-2">
            <button id="run-btn" phx-hook="RunBlockly"
                    class="px-6 py-2 rounded-lg font-bold bg-green-600 hover:bg-green-500">
              Run
            </button>
            <button id="stop-btn" phx-click="stop_program"
                    class="px-6 py-2 rounded-lg font-bold bg-red-600 hover:bg-red-500 hidden">
              Stop
            </button>
          </div>
        </div>
      </div>

      <%!-- Blockly workspace --%>
      <div class="flex" style="height: calc(100vh - 72px);">
        <div id="blockly-mount" phx-hook="Blockly" phx-update="ignore" class="flex-1 relative">
          <div id="blockly-target" style="position:absolute;inset:0;"></div>
        </div>

        <%!-- Log panel --%>
        <div id="blocks-log" class="w-64 bg-gray-800 border-l border-gray-700 p-4 overflow-y-auto">
          <h3 class="font-semibold mb-2 text-gray-200">Log</h3>
          <div id="blocks-log-entries" class="text-xs font-mono space-y-1 text-gray-300"></div>
        </div>
      </div>
    </div>
    """
  end
end
