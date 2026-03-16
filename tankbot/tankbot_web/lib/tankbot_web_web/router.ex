defmodule TankbotWebWeb.Router do
  use TankbotWebWeb, :router

  pipeline :browser do
    plug :accepts, ["html"]
    plug :fetch_session
    plug :fetch_live_flash
    plug :put_root_layout, html: {TankbotWebWeb.Layouts, :root}
    plug :protect_from_forgery
    plug :put_secure_browser_headers
  end

  pipeline :api do
    plug :accepts, ["json"]
  end

  scope "/", TankbotWebWeb do
    pipe_through :browser

    live "/", DashboardLive, :index
    live "/blocks", BlocksLive, :index
  end

  # Other scopes may use custom stacks.
  # scope "/api", TankbotWebWeb do
  #   pipe_through :api
  # end
end
