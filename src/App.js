import React, { Component } from 'react';
import { SearchBar } from './SearchBar';
import { RabbitHole } from './RabbitHole';
import './App.css';

class App extends Component {
  constructor() {
    super();
    this.state = {};
  }

  render() {
    return (
      <div className="App">
        <header className="App-header">
          <h1 className="App-title">Weirdness Engine</h1>
        </header>
        <main className="App-main">
          {!this.state.results && (
            <SearchBar onSearch={searchTerm => this.runSearch(searchTerm)} />
          )}
          {this.state.results && <RabbitHole results={this.state.results} />}
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
    /*let form = new FormData();
    form.append('text', searchTerm);
    fetch('someUrl', {
      method: 'POST',
      body: form
    })
      .then(res => res.json())
      .then(vector => this.setState({ vector }))
      .catch(err => console.error(err));*/
    setTimeout(() => {
      let results = [
        {
          id: '1',
          imageUrl:
            'https://images.unsplash.com/photo-1530794006412-5429c041040f?ixlib=rb-0.3.5&ixid=eyJhcHBfaWQiOjEyMDd9&s=d44bc6e858efc4625c6e36f7941732a2&auto=format&fit=crop&w=1950&q=80'
        },
        {
          id: '2',
          imageUrl:
            'https://images.unsplash.com/photo-1530790359200-e2cac35c770d?ixlib=rb-0.3.5&ixid=eyJhcHBfaWQiOjEyMDd9&s=a8ecadf2c46381d136651afbbe50208f&auto=format&fit=crop&w=1955&q=80'
        },
        {
          id: '3',
          imageUrl:
            'https://images.unsplash.com/photo-1530785101372-ac4eb402d2cd?ixlib=rb-0.3.5&ixid=eyJhcHBfaWQiOjEyMDd9&s=fc6cea47d594f98424f9047846c4b2cf&auto=format&fit=crop&w=1953&q=80'
        }
      ];
      this.setState({ results });
    }, 1000);
  }
}

export default App;
