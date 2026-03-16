defmodule TankbotWeb.RobotSocket do
  @moduledoc """
  WebSocket client that maintains a persistent connection to the robot.
  Receives telemetry and video frames, broadcasts them via PubSub.
  """
  use GenServer
  require Logger

  @robot_url "ws://192.168.178.45:9000"
  @reconnect_interval 3_000

  def start_link(opts \\ []) do
    url = Keyword.get(opts, :url, @robot_url)
    GenServer.start_link(__MODULE__, url, name: __MODULE__)
  end

  def send_command(command) when is_map(command) do
    GenServer.cast(__MODULE__, {:send, Jason.encode!(command)})
  end

  # Convenience functions
  def motor(left, right), do: send_command(%{cmd: "motor", left: left, right: right})
  def servo(channel, angle), do: send_command(%{cmd: "servo", channel: channel, angle: angle})
  def led(r, g, b, mask \\ 0xF), do: send_command(%{cmd: "led", r: r, g: g, b: b, mask: mask})
  def led_off(), do: send_command(%{cmd: "led_off"})
  def stop(), do: send_command(%{cmd: "stop"})
  def set_mode(mode), do: send_command(%{cmd: "mode", mode: mode})
  def arm(dir), do: send_command(%{cmd: "arm", dir: dir})
  def grabber(dir), do: send_command(%{cmd: "grabber", dir: dir})

  # --- GenServer callbacks ---

  @impl true
  def init(url) do
    send(self(), :connect)
    {:ok, %{url: url, conn: nil, connected: false}}
  end

  @impl true
  def handle_info(:connect, state) do
    case WebSockex.start_link(state.url, __MODULE__.Handler, self()) do
      {:ok, pid} ->
        Logger.info("Connected to robot at #{state.url}")
        # Subscribe to video
        WebSockex.send_frame(pid, {:text, Jason.encode!(%{type: "subscribe", channels: ["video"]})})
        {:noreply, %{state | conn: pid, connected: true}}

      {:error, reason} ->
        Logger.warning("Failed to connect to robot: #{inspect(reason)}, retrying...")
        Process.send_after(self(), :connect, @reconnect_interval)
        {:noreply, %{state | connected: false}}
    end
  end

  def handle_info({:robot_text, text}, state) do
    case Jason.decode(text) do
      {:ok, data} ->
        Phoenix.PubSub.broadcast(TankbotWeb.PubSub, "robot:telemetry", {:telemetry, data})
      _ ->
        :ok
    end
    {:noreply, state}
  end

  def handle_info({:robot_binary, frame}, state) do
    Phoenix.PubSub.broadcast(TankbotWeb.PubSub, "robot:video", {:frame, frame})
    {:noreply, state}
  end

  def handle_info({:robot_disconnected, _reason}, state) do
    Logger.warning("Robot disconnected, reconnecting...")
    Process.send_after(self(), :connect, @reconnect_interval)
    {:noreply, %{state | conn: nil, connected: false}}
  end

  @impl true
  def handle_cast({:send, payload}, %{conn: conn, connected: true} = state) when not is_nil(conn) do
    WebSockex.send_frame(conn, {:text, payload})
    {:noreply, state}
  end

  def handle_cast({:send, _payload}, state) do
    Logger.warning("Cannot send command — not connected to robot")
    {:noreply, state}
  end
end

defmodule TankbotWeb.RobotSocket.Handler do
  @moduledoc false
  use WebSockex

  def handle_frame({:text, text}, parent) do
    send(parent, {:robot_text, text})
    {:ok, parent}
  end

  def handle_frame({:binary, data}, parent) do
    send(parent, {:robot_binary, data})
    {:ok, parent}
  end

  def handle_disconnect(_conn_status, parent) do
    send(parent, {:robot_disconnected, :closed})
    {:ok, parent}
  end
end
