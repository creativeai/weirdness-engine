import React, { Component } from 'react';
import { SearchBar } from './SearchBar';
import './App.css';

class App extends Component {
  render() {
    return (
      <div className="App">
        <header className="App-header">
          <h1 className="App-title">Weirdness Engine</h1>
        </header>
        <main className="App-main">
          <SearchBar />
        </main>
        <footer className="App-footer">
          <p>&copy;{new Date().getFullYear()} Weirdness Engine &bull; <a>Privacy</a></p>
        </footer>
      </div>
    );
  }
}

export default App;