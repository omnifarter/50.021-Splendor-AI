import React, { useContext } from 'react';
import logo from './logo.svg';
import './App.css';
import Home from './page/Home';
import { QueryClient, QueryClientProvider } from 'react-query';

function App() {
  const queryClient = new QueryClient()
  return (
    <QueryClientProvider client={queryClient}>
    <div className="text-white">
        <Home />
      </div>
    </QueryClientProvider>
  );
}

export default App;
