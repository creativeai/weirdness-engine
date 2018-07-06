import React, { Component } from 'react';
import { imgSrcToBlob } from 'blob-util';

import { SearchBar } from './SearchBar';
import { RabbitHole } from './RabbitHole';
import './App.css';

// What even is this, lol.
function getCaption(item) {
  return new Promise(resolve => {
    imgSrcToBlob(item.imageUrl, 'image/jpeg', 'Anonymous')
      .then(blob => {
        let formData = new FormData();
        formData.append('image', blob, 'image.jpg');
        fetch(
          'http://ec2-52-214-48-175.eu-west-1.compute.amazonaws.com:8787/api/v1/i2c',
          {
            method: 'POST',
            body: formData
          }
        )
          .then(res => res.json())
          .then(res => {
            if (res.status === 'ok') {
              item.caption = res.response;
            }
            resolve(item);
          });
      })
      .catch(e => console.error(e));
  });
}

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
    let results = [
      {
        id: '1',
        imageUrl: 'testimages/1.jpeg'
      },
      {
        id: '2',
        imageUrl: 'testimages/2.jpeg'
      },
      {
        id: '3',
        imageUrl: 'testimages/3.jpeg'
      },
      {
        id: '4',
        imageUrl: 'testimages/4.png'
      }
    ];
    Promise.all(results.map(getCaption)).then(() => {
      this.setState({ results });
    });
  }
}

export default App;
