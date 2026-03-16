defmodule TankbotWebWeb.BlocksLive do
  @moduledoc """
  Blockly-based visual programming interface.

  Uses Google Blockly loaded as a JS hook. The workspace defines robot-specific
  blocks (drive, turn, wait, sense distance, LED, etc.) and generates a simple
  command list that gets sent to the robot via the WebSocket.
  """
  use TankbotWebWeb, :live_view

  @impl true
  def mount(_params, _session, socket) do
    if connected?(socket) do
      Phoenix.PubSub.subscribe(TankbotWeb.PubSub, "robot:telemetry")
    end

    {:ok,
     assign(socket,
       running: false,
       distance: 0.0,
       log: [],
       page_title: "Block Programming"
     )}
  end

  @impl true
  def handle_info({:telemetry, data}, socket) do
    {:noreply, assign(socket, distance: data["distance"] || 0.0)}
  end

  @impl true
  def handle_event("run_program", %{"commands" => commands}, socket) when is_list(commands) do
    # Commands come from Blockly JS as a list of maps
    Task.start(fn -> execute_program(commands) end)
    {:noreply, assign(socket, running: true, log: ["Program started..." | socket.assigns.log])}
  end

  def handle_event("stop_program", _params, socket) do
    TankbotWeb.RobotSocket.stop()
    {:noreply, assign(socket, running: false, log: ["Program stopped." | socket.assigns.log])}
  end

  defp execute_program(commands) do
    Enum.each(commands, fn cmd ->
      case cmd do
        %{"type" => "drive", "left" => left, "right" => right, "duration" => duration} ->
          TankbotWeb.RobotSocket.motor(left, right)
          Process.sleep(trunc(duration * 1000))
          TankbotWeb.RobotSocket.stop()

        %{"type" => "stop"} ->
          TankbotWeb.RobotSocket.stop()

        %{"type" => "wait", "seconds" => seconds} ->
          Process.sleep(trunc(seconds * 1000))

        %{"type" => "led", "r" => r, "g" => g, "b" => b} ->
          TankbotWeb.RobotSocket.led(r, g, b)

        %{"type" => "led_off"} ->
          TankbotWeb.RobotSocket.led_off()

        %{"type" => "servo", "channel" => ch, "angle" => angle} ->
          TankbotWeb.RobotSocket.servo(ch, angle)

        _ ->
          :ignore
      end
    end)

    TankbotWeb.RobotSocket.stop()
  end

  @impl true
  def render(assigns) do
    ~H"""
    <div class="min-h-screen bg-gray-900 text-white">
      <%!-- Header --%>
      <div class="bg-gray-800 p-4 flex items-center justify-between">
        <div class="flex items-center gap-4">
          <.link navigate={~p"/"} class="text-gray-400 hover:text-white">← Back</.link>
          <h1 class="text-2xl font-bold">🧩 Block Programming</h1>
        </div>
        <div class="flex items-center gap-4">
          <span class="text-sm text-gray-400">Distance: <%= :erlang.float_to_binary(@distance / 1, [decimals: 1]) %> cm</span>
          <%= if @running do %>
            <button phx-click="stop_program" class="bg-red-600 hover:bg-red-500 px-6 py-2 rounded-lg font-bold">
              ⏹ Stop
            </button>
          <% else %>
            <button id="run-btn" phx-hook="RunBlockly" class="bg-green-600 hover:bg-green-500 px-6 py-2 rounded-lg font-bold">
              ▶ Run
            </button>
          <% end %>
        </div>
      </div>

      <%!-- Blockly workspace --%>
      <div class="flex" style="height: calc(100vh - 72px);">
        <div id="blockly-workspace" phx-hook="Blockly" phx-update="ignore" class="flex-1"></div>

        <%!-- Log panel --%>
        <div class="w-64 bg-gray-800 p-4 overflow-y-auto">
          <h3 class="font-semibold mb-2">Log</h3>
          <div class="text-xs font-mono space-y-1 text-gray-400">
            <%= for entry <- Enum.take(@log, 50) do %>
              <div><%= entry %></div>
            <% end %>
          </div>
        </div>
      </div>
    </div>
    """
  end
end
