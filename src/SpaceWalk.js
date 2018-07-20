import React, { Component } from 'react';
import _ from 'lodash';

import { SpaceWalkPath } from './SpaceWalkPath';

import './SpaceWalk.css';

export class SpaceWalk extends Component {
  constructor() {
    super();
    this.state = {
      fromTerm: '',
      toTerm: '',
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
          <input
            type="search"
            className="search left"
            value={this.state.fromTerm}
            onInput={e => this.setState({ fromTerm: e.target.value })}
          />
          <input
            type="search"
            className="search right"
            value={this.state.toTerm}
            onInput={e => this.setState({ toTerm: e.target.value })}
          />
          <input
            type="submit"
            value="Take a walk"
            className="button"
            disabled={
              _.isEmpty(this.state.fromTerm) || _.isEmpty(this.state.toTerm)
            }
          />
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
    let n = 50;
    let generateNext = () => {
      this.setState({
        items: [
          ...this.state.items,
          {
            position: Math.random(),
            size: 0.3 + Math.random() * 0.7,
            xOffset: _.random(-100, 100),
            yOffset: _.random(-100, 100),
            url: _.sample([
              'testimages/1.jpeg',
              'testimages/2.jpeg',
              'testimages/3.jpeg',
              'testimages/4.png'
            ])
          }
        ]
      });
      if (n-- > 0) setTimeout(generateNext, Math.random() * 100);
    };
    setTimeout(generateNext, 1000);
  }
}
