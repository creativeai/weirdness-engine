import React, { Component } from 'react';
import { imgSrcToBlob } from 'blob-util';

import { SearchBar } from './SearchBar';
import { RabbitHole } from './RabbitHole';
import { TreasureMap } from './TreasureMap';
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

function getSimilarImages(item) {
  return new Promise(resolve => {
    imgSrcToBlob(item.imageUrl, 'image/jpeg', 'Anonymous')
      .then(blob => {
        let formData = new FormData();
        formData.append('image', blob, 'image.jpg');
        fetch(
          'http://ec2-34-243-15-197.eu-west-1.compute.amazonaws.com:5000/api/v1/img_search',
          {
            method: 'POST',
            body: formData
          }
        )
          .then(res => res.json())
          .then(res => {
            if (res.status === 'ok') {
              resolve(
                res.response.map(imageUrl => ({
                  imageUrl: `http://ec2-34-243-15-197.eu-west-1.compute.amazonaws.com:5000/${imageUrl}`
                }))
              );
            }
          });
      })
      .catch(e => console.error(e));
  });
}

class App extends Component {
  constructor() {
    super();
    this.state = {
      treasureMapItems: [
        {
          id: '1',
          imageUrl: 'testimages/1.jpeg',
          x: 0,
          y: 0
        },
        {
          id: '2',
          imageUrl: 'testimages/2.jpeg',
          x: 1,
          y: 1
        },
        {
          id: '3',
          imageUrl: 'testimages/3.jpeg',
          x: Math.random(),
          y: Math.random()
        },
        {
          id: '4',
          imageUrl: 'testimages/4.png',
          x: Math.random(),
          y: Math.random()
        }
      ]
    };
  }

  render() {
    return (
      <div className="App">
        <main className="App-main">
          <TreasureMap items={this.state.treasureMapItems} />
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

  componentWillMount() {
    getSimilarImages(this.state.treasureMapItems[0])
      .then(similarImages => {
        this.setState({
          treasureMapItems: this.state.treasureMapItems.concat(
            similarImages.map(res => ({
              ...res,
              x: Math.random() * 0.25,
              y: Math.random() * 0.25
            }))
          )
        });
      })
      .then(() => {
        getSimilarImages(
          this.state.treasureMapItems[this.state.treasureMapItems.length - 1]
        ).then(similarImages => {
          this.setState({
            treasureMapItems: this.state.treasureMapItems.concat(
              similarImages.map(res => ({
                ...res,
                x: 0.75 + Math.random() * 0.25,
                y: 0.75 + Math.random() * 0.25
              }))
            )
          });
        });
      });
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
