import React, { useContext } from 'react';
import logo from './logo.svg';
import './App.css';
import Home from './page/Home';
import {GameContext, useGameProvider} from './helpers/useGameProvider'

function App() {
  const gameProvider = useGameProvider()
  return (
    <GameContext.Provider value={gameProvider}>
      <div className="text-white">
        <Home />
      </div>
    </GameContext.Provider>
  );
}

export default App;
