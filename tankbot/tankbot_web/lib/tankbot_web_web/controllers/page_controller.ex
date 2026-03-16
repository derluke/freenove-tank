defmodule TankbotWebWeb.PageController do
  use TankbotWebWeb, :controller

  def home(conn, _params) do
    render(conn, :home)
  end
end
