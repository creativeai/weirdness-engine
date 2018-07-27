import React, { Component } from 'react';
import EventListener from 'react-event-listener';

import './Viewport.css';

const MIN_SCALE = 0.5;
const MAX_SCALE = 5;

export class Viewport extends Component {
  constructor() {
    super();
    this.state = {
      zoomLevel: 1,
      translate: [0, 0],
      panning: false,
      lastPanPoint: null
    };
  }

  render() {
    return (
      <div
        className="viewport"
        onWheel={e => this.onWheel(e)}
        onMouseDown={e => this.onStartPan(e)}
        style={{
          transform: `scale(${this.state.zoomLevel}) translate(${
            this.state.translate[0]
          }px, ${this.state.translate[1]}px)`
        }}
      >
        <EventListener
          target="document"
          onMouseMove={e => this.onPan(e)}
          onMouseUp={e => this.onStopPan(e)}
        />
        {this.props.children}
      </div>
    );
  }

  onWheel(evt) {
    let changeFactor = 1 + -evt.deltaY / 30;
    let newZoom = Math.min(
      MAX_SCALE,
      Math.max(MIN_SCALE, this.state.zoomLevel * changeFactor)
    );
    this.setState({ zoomLevel: newZoom });
    evt.preventDefault();
  }

  onStartPan(evt) {
    this.setState({ panning: true, lastPanPoint: [evt.clientX, evt.clientY] });
    evt.preventDefault();
  }

  onStopPan(evt) {
    if (this.state.panning) {
      this.setState({ panning: false, lastPanPoint: null });
      evt.preventDefault();
    }
  }

  onPan(evt) {
    if (this.state.panning) {
      let [lastX, lastY] = this.state.lastPanPoint;
      let deltaX = (evt.clientX - lastX) / this.state.zoomLevel;
      let deltaY = (evt.clientY - lastY) / this.state.zoomLevel;
      this.setState({
        lastPanPoint: [evt.clientX, evt.clientY],
        translate: [
          this.state.translate[0] + deltaX,
          this.state.translate[1] + deltaY
        ]
      });
    }
  }
}
