import React, { Component } from 'react';
import _ from 'lodash';

import { isoLines } from 'marchingsquares';
import { simplify } from './simplify';
import * as turf from '@turf/turf';
import polygonSmooth from '@turf/polygon-smooth';

import './SpaceWalkContours.css';

const GRADIENT_SCALE_FACTOR = 7;
const SCALEDOWN = 20;

export class SpaceWalkContours extends Component {
  constructor() {
    super();
    this.canvasRef = React.createRef();
    this.state = {};
  }

  componentDidMount() {
    this.renderCanvas();
  }

  componentDidUpdate() {
    this.renderCanvas();
  }

  render() {
    return <canvas className="spaceWalkContours" ref={this.canvasRef} />;
  }

  renderCanvas() {
    let cnvs = this.canvasRef.current;
    let width = (cnvs.width = cnvs.offsetWidth);
    let height = (cnvs.height = cnvs.offsetHeight);
    let scaleDownWidth = Math.round(width / SCALEDOWN);
    let scaleDownHeight = Math.round(height / SCALEDOWN);

    let ctx = cnvs.getContext('2d');
    ctx.clearRect(0, 0, width, height);

    // Draw gradients for each item (used as contour basis, not actually shown)

    let gradientCnvs = document.createElement('canvas');
    let gradientCtx = gradientCnvs.getContext('2d');
    gradientCnvs.width = scaleDownWidth;
    gradientCnvs.height = scaleDownHeight;
    gradientCtx.fillStyle = 'black';
    gradientCtx.fillRect(0, 0, scaleDownWidth, scaleDownHeight);
    gradientCtx.globalCompositeOperation = 'lighter';

    for (let box of this.props.itemBoxes) {
      let cntrX = (box.left + box.width / 2) / SCALEDOWN;
      let cntrY = (box.top + box.height / 2) / SCALEDOWN;
      let grad = gradientCtx.createRadialGradient(
        cntrX,
        cntrY,
        0,
        cntrX,
        cntrY,
        (box.width * GRADIENT_SCALE_FACTOR) / SCALEDOWN
      );
      grad.addColorStop(0, 'rgba(19, 19, 19, 1)');
      grad.addColorStop(1, 'rgba(19, 19, 19, 0)');
      gradientCtx.fillStyle = grad;
      gradientCtx.beginPath();
      gradientCtx.arc(
        cntrX,
        cntrY,
        (box.width * GRADIENT_SCALE_FACTOR) / SCALEDOWN,
        0,
        Math.PI * 2
      );
      gradientCtx.fill();
    }

    // Measure contours
    let zData = _.range(scaleDownHeight).map(() => new Array(scaleDownWidth));
    let imageData = gradientCtx.getImageData(
      0,
      0,
      scaleDownWidth,
      scaleDownHeight
    ).data;
    let channelCount = imageData.length / (scaleDownWidth * scaleDownHeight);
    for (let x = 0; x < scaleDownWidth; x++) {
      for (let y = 0; y < scaleDownHeight; y++) {
        let redIdx = y * (scaleDownWidth * channelCount) + x * channelCount;
        let red = imageData[redIdx];
        zData[y][x] = 255 - red;
      }
    }
    let levels = isoLines(zData, [
      240,
      220,
      200,
      180,
      160,
      140,
      120,
      100,
      80,
      60,
      40,
      20
    ]);

    // Draw contour lines
    ctx.globalCompositeOperation = 'source-over';
    ctx.strokeStyle = 'rgb(180, 180, 180)';
    ctx.lineWidth = 0.5;

    let outermostPolygon;
    for (let bands of levels) {
      if (bands.length === 1 && bands[0].length === 5) {
        // It's nothing, just the bounding rect
        continue;
      }
      for (let band of bands) {
        if (band.length === 5) {
          continue;
        }
        let sBand = simplify(band, 0.75);
        if (sBand.length >= 4) {
          let polygon = turf.polygon([sBand]);
          let smoothed = polygonSmooth(polygon, { iterations: 2 });
          sBand = smoothed.features[0].geometry.coordinates[0];
        }

        ctx.beginPath();
        ctx.moveTo(sBand[0][0] * SCALEDOWN, sBand[0][1] * SCALEDOWN);
        for (let i = 1; i < sBand.length; i++) {
          ctx.lineTo(sBand[i][0] * SCALEDOWN, sBand[i][1] * SCALEDOWN);
        }

        ctx.stroke();
      }
    }
  }
}
