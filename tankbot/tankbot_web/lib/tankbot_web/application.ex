defmodule TankbotWeb.Application do
  # See https://hexdocs.pm/elixir/Application.html
  # for more information on OTP Applications
  @moduledoc false

  use Application

  @impl true
  def start(_type, _args) do
    children = [
      TankbotWebWeb.Telemetry,
      {DNSCluster, query: Application.get_env(:tankbot_web, :dns_cluster_query) || :ignore},
      {Phoenix.PubSub, name: TankbotWeb.PubSub},
      # Robot WebSocket client — connects to the Pi
      TankbotWeb.RobotSocket,
      # Start to serve requests, typically the last entry
      TankbotWebWeb.Endpoint
    ]

    # See https://hexdocs.pm/elixir/Supervisor.html
    # for other strategies and supported options
    opts = [strategy: :one_for_one, name: TankbotWeb.Supervisor]
    Supervisor.start_link(children, opts)
  end

  # Tell Phoenix to update the endpoint configuration
  # whenever the application is updated.
  @impl true
  def config_change(changed, _new, removed) do
    TankbotWebWeb.Endpoint.config_change(changed, removed)
    :ok
  end
end
