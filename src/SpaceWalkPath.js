import React, { Component } from 'react';
import { CSSTransition, TransitionGroup } from 'react-transition-group';

import { toPath } from 'svg-catmull-rom-spline';
import _ from 'lodash';

import { SpaceWalkTerrain } from './SpaceWalkTerrain';
import { SpaceWalkContours } from './SpaceWalkContours';

import './SpaceWalkPath.css';

const PATH_MARGIN_VERTICAL = 100;
const PATH_MARGIN_HORIZONTAL = 150;
const TOP_LEFT_CORNER = [PATH_MARGIN_HORIZONTAL, PATH_MARGIN_VERTICAL];
const BOTTOM_RIGHT_CORNER = [
  window.innerWidth - PATH_MARGIN_HORIZONTAL,
  window.innerHeight - PATH_MARGIN_VERTICAL
];
const MAX_ITEM_SIZE = 50;

export class SpaceWalkPath extends Component {
  constructor() {
    super();
    this.toTopLeftInitialRef = React.createRef();
    this.toTopLeftExtendedRef = React.createRef();
    this.toBottomRightInitialRef = React.createRef();
    this.toBottomRightExtendedRef = React.createRef();
    console.log('tli');
    let toTopLeftInitialPoints = generatePath(
      [
        {
          startPoint: [window.innerWidth / 2, window.innerHeight / 2],
          endPoint: [window.innerWidth / 2 - 100, window.innerHeight / 2],
          maxPoints: 10
        }
      ],
      20
    );
    console.log('tle');
    let toTopLeftExtendedPoints = generatePath(
      generateLeftPathQuadrants(
        [window.innerWidth / 2 - 100, window.innerHeight / 2],
        TOP_LEFT_CORNER
      ),
      50,
      getHeading(..._.takeRight(toTopLeftInitialPoints, 2))
    );
    console.log('tri');
    let toBottomRightInitialPoints = generatePath(
      [
        {
          startPoint: [window.innerWidth / 2, window.innerHeight / 2],
          endPoint: [window.innerWidth / 2 + 100, window.innerHeight / 2],
          maxPoints: 10
        }
      ],
      20
    );
    console.log('bre');
    let toBottomRightExtendedPoints = generatePath(
      generateRightPathQuadrants(
        [window.innerWidth / 2 + 100, window.innerHeight / 2],
        BOTTOM_RIGHT_CORNER
      ),
      50,
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
    let items = this.props.items.map(item => this.getItemBox(item));
    return (
      <div className="spaceWalkPathWrap">
        {/*<SpaceWalkTerrain itemBoxes={items} />*/}
        <svg
          className="spaceWalkPath"
          viewBox={`0 0 ${window.innerWidth} ${window.innerHeight}`}
        >
          <SpaceWalkContours
            itemBoxes={items}
            width={window.innerWidth}
            height={window.innerHeight}
          />
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
        </svg>
        <div className="spaceWalkPathItems">
          <TransitionGroup className="item-list" component={null}>
            {items.map((item, index) => (
              <CSSTransition key={index} timeout={500} classNames="fade">
                <img
                  key={index}
                  className="item"
                  src={item.url}
                  style={{
                    left: item.left,
                    top: item.top,
                    width: item.width,
                    height: item.height
                  }}
                />
              </CSSTransition>
            ))}
          </TransitionGroup>
        </div>
      </div>
    );
  }

  getItemBox({ url, position, size, xOffset, yOffset }) {
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
        this.state.toTopLeftExtendedLength - positionAlongLength
      );
    } else if (
      positionAlongLength <
      this.state.toTopLeftExtendedLength + this.state.toTopLeftInitialLength
    ) {
      let dist =
        this.state.toTopLeftInitialLength -
        (positionAlongLength - this.state.toTopLeftExtendedLength);
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
      url,
      left: pt.x - pxSize + xOffset,
      top: pt.y - pxSize + yOffset,
      width: pxSize,
      height: pxSize
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

function generateLeftPathQuadrants(startPoint, endPoint) {
  let tries = 0;

  while (true) {
    try {
      let result = [];
      let mid1 = [window.innerWidth / 4, window.innerHeight / 5];
      let mid2 = [window.innerWidth / 3, (window.innerHeight * 4) / 5];
      result.push({
        startPoint: startPoint,
        endPoint: mid1,
        maxPoints: 20
      });
      result.push({
        startPoint: mid1,
        endPoint: mid2,
        maxPoints: 20
      });
      result.push({
        startPoint: mid2,
        endPoint: endPoint,
        maxPoints: 20
      });
      return result;
    } catch (e) {
      if (tries++ > 20) {
        throw e;
      }
    }
  }
}

function generateRightPathQuadrants(startPoint, endPoint) {
  let tries = 0;

  while (true) {
    try {
      let result = [];
      let mid1 = [(window.innerWidth * 3) / 4, (window.innerHeight * 4) / 5];
      let mid2 = [(window.innerWidth * 2) / 3, window.innerHeight / 5];
      result.push({
        startPoint: startPoint,
        endPoint: mid1,
        maxPoints: 20
      });
      result.push({
        startPoint: mid1,
        endPoint: mid2,
        maxPoints: 20
      });
      result.push({
        startPoint: mid2,
        endPoint: endPoint,
        maxPoints: 20
      });
      return result;
    } catch (e) {
      if (tries++ > 20) {
        throw e;
      }
    }
  }
}

function generatePath(
  segments,
  stepLength,
  startHeading = getHeading(segments[0].startPoint, segments[0].endPoint)
) {
  let result = [];

  let heading = startHeading;
  for (let { startPoint, endPoint, maxPoints } of segments) {
    let tries = 0,
      resultSegment;
    while (tries++ < 100) {
      let attempt = tryGeneratePath(startPoint, endPoint, stepLength, heading);
      if (attempt.length < maxPoints) {
        resultSegment = attempt;
        heading = getHeading(..._.takeRight(resultSegment, 2));
      }
    }
    if (resultSegment) {
      result = result.concat(resultSegment);
    } else {
      throw 'Could not find short enough path';
    }
  }
  return result;
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
