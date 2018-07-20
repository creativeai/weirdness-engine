import React, { Component } from 'react';
import _ from 'lodash';

import { SpaceWalkPath } from './SpaceWalkPath';
import { quantizeKMeans } from './kmeans';

import './SpaceWalk.css';

export class SpaceWalk extends Component {
  constructor() {
    super();
    this.state = {
      extended: false,
      items: []
    };
  }

  render() {
    return (
      <div className={`spaceWalk ${this.state.extended ? 'is-extended' : ''}`}>
        <SpaceWalkPath
          extended={this.state.extended}
          items={this.state.items}
        />
        <form onSubmit={e => this.onExtend(e)}>
          <input type="search" className="search left" />
          <input type="search" className="search right" />
          <input type="submit" value="Take a walk" className="button" />
        </form>
      </div>
    );
  }

  onExtend(evt) {
    this.setState({ extended: true });
    this.generateItems();
    evt.preventDefault();
  }

  generateItems() {
    let itemsPromise = Promise.all(
      _.range(50).map(() => {
        let imgUrl = _.sample([
          'testimages/1.jpeg',
          'testimages/2.jpeg',
          'testimages/3.jpeg',
          'testimages/4.png'
        ]);
        return this.getDominantColor(imgUrl).then(color => {
          return {
            position: Math.random(),
            size: 0.3 + Math.random() * 0.7,
            xOffset: _.random(-100, 100),
            yOffset: _.random(-100, 100),
            url: imgUrl,
            color
          };
        });
      })
    );
    itemsPromise.then(items => {
      let generateNext = () => {
        this.setState({
          items: [...this.state.items, items.shift()]
        });
        if (items.length) setTimeout(generateNext, 100);
      };
      setTimeout(generateNext, 0);
    });
  }

  getDominantColor(imgUrl) {
    return new Promise(res => {
      let img = document.createElement('img');
      img.src = imgUrl;
      img.onload = () => {
        res(quantizeKMeans(img));
      };
    });
  }
}
