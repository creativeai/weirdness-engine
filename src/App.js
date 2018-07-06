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
          <SearchBar onSearch={searchTerm => this.runSearch(searchTerm)} />
        </main>
        <footer className="App-footer">
          <p>
            &copy;{new Date().getFullYear()} Weirdness Engine &bull;{' '}
            <a>Privacy</a>
          </p>
        </footer>
      </div>
    );
  }

  runSearch(searchTerm) {
    let form = new FormData();
    form.append('text', searchTerm);
    fetch('someUrl', {
      method: 'POST',
      body: form
    })
      .then(res => res.json())
      .then(vector => this.setState({ vector }))
      .catch(err => console.error(err));
  }
}

export default App;
