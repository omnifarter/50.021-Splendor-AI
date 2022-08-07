import React, { useContext } from "react";
import logo from "./logo.svg";
import "./App.css";
import Home from "./page/Home";
import { QueryClient, QueryClientProvider } from "react-query";
import { NotificationsProvider } from "@mantine/notifications";
import { MantineProvider } from "@mantine/core";

function App() {
  const queryClient = new QueryClient();
  return (
    <MantineProvider>
      <NotificationsProvider>
        <QueryClientProvider client={queryClient}>
          <div className="text-white">
            <Home />
          </div>
        </QueryClientProvider>
      </NotificationsProvider>
    </MantineProvider>
  );
}

export default App;
