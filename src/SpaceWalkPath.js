import React, { Component } from 'react';
import { CSSTransition, TransitionGroup } from 'react-transition-group';

import { toPath } from 'svg-catmull-rom-spline';
import _ from 'lodash';
import './SpaceWalkPath.css';

const PATH_MARGIN_VERTICAL = 50;
const PATH_MARGIN_HORIZONTAL = 100;
const TOP_LEFT_CORNER = [PATH_MARGIN_HORIZONTAL, PATH_MARGIN_VERTICAL];
const BOTTOM_RIGHT_CORNER = [
  window.innerWidth - PATH_MARGIN_HORIZONTAL,
  window.innerHeight - PATH_MARGIN_VERTICAL
];
const MAX_ITEM_SIZE = 50;
const ITEM_HALO_SCALE_FACTOR = 7;

export class SpaceWalkPath extends Component {
  constructor() {
    super();
    this.toTopLeftInitialRef = React.createRef();
    this.toTopLeftExtendedRef = React.createRef();
    this.toBottomRightInitialRef = React.createRef();
    this.toBottomRightExtendedRef = React.createRef();
    let toTopLeftInitialPoints = generatePath(
      [window.innerWidth / 2, window.innerHeight / 2],
      [window.innerWidth / 2 - 100, window.innerHeight / 2],
      20,
      10
    );
    let toTopLeftExtendedPoints = generatePath(
      [window.innerWidth / 2 - 100, window.innerHeight / 2],
      TOP_LEFT_CORNER,
      50,
      25,
      getHeading(..._.takeRight(toTopLeftInitialPoints, 2))
    );
    let toBottomRightInitialPoints = generatePath(
      [window.innerWidth / 2, window.innerHeight / 2],
      [window.innerWidth / 2 + 100, window.innerHeight / 2],
      20,
      10
    );
    let toBottomRightExtendedPoints = generatePath(
      [window.innerWidth / 2 + 100, window.innerHeight / 2],
      BOTTOM_RIGHT_CORNER,
      50,
      25,
      getHeading(..._.takeRight(toBottomRightInitialPoints, 2))
    );

    this.state = {
      toTopLeftInitialPoints,
      toTopLeftExtendedPoints,
      toTopLeftInitialLength: 0,
      toTopLeftExtendedLength: 0,
      toBottomRightInitialLength: 0,
      toBottomRightExtendedLength: 0,
      toBottomRightInitialPoints,
      toBottomRightExtendedPoints
    };
  }

  render() {
    let items = this.props.items.map((item, index) =>
      this.renderItem(item, index)
    );
    return (
      <div className="spaceWalkPathWrap">
        <svg
          className="spaceWalkPath"
          viewBox={`0 0 ${window.innerWidth} ${window.innerHeight}`}
        >
          <defs>
            <radialGradient id="itemHalo">
              <stop offset="10%" stop-color="rgba(0, 0, 0, 0)" />
              <stop offset="95%" stop-color="rgba(0, 0, 0, 0)" />
            </radialGradient>
          </defs>
          <path
            ref={this.toTopLeftInitialRef}
            className="path initial"
            d={generateCurve(this.state.toTopLeftInitialPoints)}
          />
          <path
            ref={this.toTopLeftExtendedRef}
            className={`path extended ${this.props.extended ? 'visible' : ''}`}
            d={generateCurve(this.state.toTopLeftExtendedPoints)}
            strokeDasharray={this.state.toTopLeftExtendedLength}
            strokeDashoffset={
              this.props.extended ? 0 : this.state.toTopLeftExtendedLength
            }
          />
          <path
            ref={this.toBottomRightInitialRef}
            className="path initial"
            d={generateCurve(this.state.toBottomRightInitialPoints)}
          />
          <path
            ref={this.toBottomRightExtendedRef}
            className={`path extended ${this.props.extended ? 'visible' : ''}`}
            d={generateCurve(this.state.toBottomRightExtendedPoints)}
            strokeDasharray={this.state.toBottomRightExtendedLength}
            strokeDashoffset={
              this.props.extended ? 0 : this.state.toBottomRightExtendedLength
            }
          />
          <TransitionGroup className="halo-list" component={null}>
            {items.map((item, index) => (
              <CSSTransition key={index} timeout={500} classNames="fade">
                {item.halo}
              </CSSTransition>
            ))}
          </TransitionGroup>
        </svg>
        <div className="spaceWalkPathItems">
          <TransitionGroup className="item-list" component={null}>
            {items.map((item, index) => (
              <CSSTransition key={index} timeout={500} classNames="fade">
                {item.item}
              </CSSTransition>
            ))}
          </TransitionGroup>
        </div>
      </div>
    );
  }

  renderItem({ url, position, size, xOffset, yOffset }, index) {
    let pxSize = size * MAX_ITEM_SIZE;
    let totalPathLength =
      this.state.toTopLeftExtendedLength +
      this.state.toTopLeftInitialLength +
      this.state.toBottomRightInitialLength +
      this.state.toBottomRightExtendedLength;
    let positionAlongLength = position * totalPathLength;
    let pt;
    if (positionAlongLength < this.state.toTopLeftExtendedLength) {
      pt = this.toTopLeftExtendedRef.current.getPointAtLength(
        positionAlongLength
      );
    } else if (
      positionAlongLength <
      this.state.toTopLeftExtendedLength + this.state.toTopLeftInitialLength
    ) {
      let dist = positionAlongLength - this.state.toTopLeftExtendedLength;
      pt = this.toTopLeftInitialRef.current.getPointAtLength(dist);
    } else if (
      positionAlongLength <
      this.state.toTopLeftExtendedLength +
        this.state.toTopLeftInitialLength +
        this.state.toBottomRightInitialLength
    ) {
      let dist =
        positionAlongLength -
        this.state.toTopLeftExtendedLength -
        this.state.toTopLeftInitialLength;
      pt = this.toBottomRightInitialRef.current.getPointAtLength(dist);
    } else {
      let dist =
        positionAlongLength -
        this.state.toTopLeftExtendedLength -
        this.state.toTopLeftInitialLength -
        this.state.toBottomRightInitialLength;
      pt = this.toBottomRightExtendedRef.current.getPointAtLength(dist);
    }
    return {
      item: (
        <img
          key={index}
          className="item"
          src={url}
          style={{
            left: pt.x - pxSize + xOffset,
            top: pt.y - pxSize + yOffset,
            width: pxSize,
            height: pxSize
          }}
        />
      ),
      halo: (
        <circle
          key={index}
          className="halo"
          fill="url(#itemHalo)"
          cx={pt.x - pxSize + xOffset}
          cy={pt.y - pxSize + yOffset}
          r={pxSize * ITEM_HALO_SCALE_FACTOR}
        />
      )
    };
  }

  componentDidMount() {
    this.setState({
      toTopLeftInitialLength: this.toTopLeftInitialRef.current.getTotalLength(),
      toTopLeftExtendedLength: this.toTopLeftExtendedRef.current.getTotalLength(),
      toBottomRightInitialLength: this.toBottomRightInitialRef.current.getTotalLength(),
      toBottomRightExtendedLength: this.toBottomRightExtendedRef.current.getTotalLength()
    });
  }
}

function generatePath(
  startPoint,
  endPoint,
  stepLength,
  maxPoints,
  startHeading = getHeading(startPoint, endPoint)
) {
  let tries = 0;
  while (tries++ < 100) {
    let attempt = tryGeneratePath(
      startPoint,
      endPoint,
      stepLength,
      startHeading
    );
    if (attempt.length < maxPoints) {
      return attempt;
    }
  }
  throw 'Could not find short enough path';
}

function tryGeneratePath(startPoint, endPoint, stepLength, startHeading) {
  let points = [startPoint];
  let n = 0,
    heading = startHeading;
  while (getDistance(_.last(points), endPoint) > 50 && n < 1000) {
    let next = generatePoint(
      _.last(points),
      endPoint,
      stepLength,
      heading,
      points.length === 1 ? 0 : 2
    );
    points.push(next.point);
    heading = next.heading;
    n++;
  }
  points.push(endPoint);
  return points;
}

function generatePoint(
  fromPoint,
  towardPoint,
  stepLength,
  heading = getHeading(fromPoint, towardPoint),
  randomAngleWeight = 2
) {
  let tries = 0,
    headingToEnd = getHeading(fromPoint, towardPoint);
  while (tries++ < 100) {
    let randomAngle = Math.random() * randomAngleWeight - randomAngleWeight / 2;
    let nextHeading =
      heading +
      (((headingToEnd - heading) * randomAngleWeight) / 6) * Math.random() +
      randomAngle;
    let distance = stepLength + Math.random() * stepLength;
    let [fromX, fromY] = fromPoint;
    let toX = fromX + Math.cos(nextHeading) * distance;
    let toY = fromY + Math.sin(nextHeading) * distance;
    if (
      toX > PATH_MARGIN_HORIZONTAL &&
      toX < window.innerWidth - PATH_MARGIN_HORIZONTAL &&
      toY > PATH_MARGIN_VERTICAL &&
      toY < window.innerHeight - PATH_MARGIN_VERTICAL
    ) {
      return { point: [toX, toY], heading: nextHeading };
    } else {
      randomAngleWeight += 0.1;
    }
  }
  throw `could not generate point from ${fromPoint} to ${towardPoint}`;
}

function getHeading(from, to) {
  return Math.atan2(to[1] - from[1], to[0] - from[0]);
}

function generateCurve(points) {
  var tolerance = 4;
  var highestQuality = true;
  var attribute = toPath(points, tolerance, highestQuality);
  return attribute;
}

function getDistance([x1, y1], [x2, y2]) {
  return Math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2);
}
