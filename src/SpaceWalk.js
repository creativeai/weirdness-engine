import React, { Component } from 'react';
import { SpaceWalkPath } from './SpaceWalkPath';

import './SpaceWalk.css';

export class SpaceWalk extends Component {
  constructor() {
    super();
    this.state = {
      extended: false
    };
  }

  render() {
    return (
      <div className={`spaceWalk ${this.state.extended ? 'is-extended' : ''}`}>
        <SpaceWalkPath extended={this.state.extended} />
        <form onSubmit={e => this.onExtend(e)}>
          <input type="search" className="search left" />
          <input type="search" className="search right" />
          <input type="submit" value="Take a walk" className="button" />
        </form>
      </div>
    );
  }

  onExtend(evt) {
    console.log('wat');
    this.setState({ extended: true });
    evt.preventDefault();
  }
}
