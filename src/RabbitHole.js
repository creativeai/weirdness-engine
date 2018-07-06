import React, { Component } from 'react';

export class RabbitHole extends Component {
  render() {
    return (
      <div className="rabbitHole">
        {this.props.results.map(result => (
          <img key={result.id} src={result.imageUrl} width="200" />
        ))}
      </div>
    );
  }
}
