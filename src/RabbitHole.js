import React, { Component } from 'react';

export class RabbitHole extends Component {
  render() {
    return (
      <div className="rabbitHole">
        {this.props.results.map(result => (
          <div key={result.id} style={{ color: 'white', display: 'flex' }}>
            <div>
              <img src={result.imageUrl} width="200" />
            </div>
            <div>{result.caption}</div>
          </div>
        ))}
      </div>
    );
  }
}
