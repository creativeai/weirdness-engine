import React, { Component } from 'react';
import _ from 'lodash';

import { SpaceWalkPath } from './SpaceWalkPath';

import './SpaceWalk.css';

function text2Vec(text) {
  let formData = new FormData();
  formData.append('text', text);
  return fetch(
    'http://ec2-34-243-15-197.eu-west-1.compute.amazonaws.com:8787/api/v1/text2vec',
    {
      method: 'POST',
      body: formData
    }
  )
    .then(res => res.json())
    .then(res => JSON.parse(res.response)[0]);
}

function vecPath(v1, v2, num_hops = 50) {
  return fetch(
    'http://ec2-34-243-15-197.eu-west-1.compute.amazonaws.com:8787/api/v1/path',
    {
      method: 'POST',
      body: JSON.stringify({ v1, v2, num_hops }),
      headers: {
        'Content-Type': 'application/json'
      }
    }
  )
    .then(res => res.json())
    .then(res => res.response);
}

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
            value={this.state.extending ? 'Generating...' : 'Take a walk'}
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
    this.setState({ extending: true });
    this.generateItems().then(items => {
      this.setState({
        extending: false,
        extended: true
      });
      items = _.shuffle(items);
      let trickleNext = () => {
        this.setState({
          items: [...this.state.items, items.shift()]
        });
        if (items.length) setTimeout(trickleNext, Math.random() * 100);
      };
      setTimeout(trickleNext, 0);
    });
    evt.preventDefault();
  }

  generateItems() {
    return text2Vec(this.state.fromTerm).then(fromVec =>
      text2Vec(this.state.toTerm).then(toVec =>
        vecPath(fromVec, toVec).then(vecPath => {
          return vecPath.map(({ path, distance }) => ({
            url: `http://ec2-34-243-15-197.eu-west-1.compute.amazonaws.com:8787/${path}`,
            position: distance,
            size: 0.3 + Math.random() * 0.7,
            xOffset: _.random(-100, 100),
            yOffset: _.random(-100, 100)
          }));
        })
      )
    );
    /*let n = 50;
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
    setTimeout(generateNext, 1000);*/
  }
}
