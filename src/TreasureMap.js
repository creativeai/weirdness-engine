import React, { Component } from 'react';

import './TreasureMap.css';

export class TreasureMap extends Component {
  render() {
    return (
      <div className="treasureMap">
        <div className="treasureMap--content">
          {this.props.items.map((item, idx) => (
            <div
              key={idx}
              className="treasureMap--item"
              style={this.getItemStyle(item)}
            >
              <img src={item.imageUrl} />
            </div>
          ))}
        </div>
      </div>
    );
  }

  getItemStyle(item) {
    return {
      left: `calc(${item.x * 100}%)`,
      top: `calc(${item.y * 100}%)`
    };
  }
}
