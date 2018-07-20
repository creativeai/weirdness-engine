import React, { Component } from 'react';
import _ from 'lodash';
import Voronoi from 'voronoi';
import * as turf from '@turf/turf';

import './SpaceWalkTerrain.css';

const ITEM_BOX_PADDING = 0;

export class SpaceWalkTerrain extends Component {
  constructor() {
    super();
    this.state = { polygons: [], itemPolys: [] };
  }

  componentWillMount() {
    let voronoi = new Voronoi();
    let bbox = { xl: 0, xr: window.innerWidth, yt: 0, yb: window.innerHeight };
    let sites = _.range(500).map(() => ({
      x: _.random(0, window.innerWidth),
      y: _.random(0, window.innerHeight)
    }));
    let diagram = voronoi.compute(sites, bbox);
    this.setState({
      polygons: diagram.cells.map(cell =>
        cell.halfedges.map(he => [he.getStartpoint(), he.getEndpoint()])
      )
    });
  }

  componentWillReceiveProps(newProps) {
    this.setState({
      itemPolys: newProps.itemBoxes.map(item =>
        turf.polygon([
          [
            [item.left - ITEM_BOX_PADDING, item.top - ITEM_BOX_PADDING],
            [
              item.left + item.width + ITEM_BOX_PADDING,
              item.top - ITEM_BOX_PADDING
            ],
            [
              item.left + item.width + ITEM_BOX_PADDING,
              item.top + item.height + ITEM_BOX_PADDING
            ],
            [
              item.left - ITEM_BOX_PADDING,
              item.top + item.height + ITEM_BOX_PADDING
            ],
            [item.left - ITEM_BOX_PADDING, item.top - ITEM_BOX_PADDING]
          ]
        ])
      )
    });
  }

  render() {
    return (
      <svg
        className="spaceWalkTerrain"
        viewBox={`0 0 ${window.innerWidth} ${window.innerHeight}`}
      >
        {this.state.polygons
          .map(poly => ({
            poly,
            intersections: this.getIntersectionCount(poly)
          }))
          .filter(({ intersections }) => intersections > 0)
          .map(({ poly, intersections }, idx) => (
            <polygon
              key={idx}
              points={poly.map(e => `${e[0].x},${e[0].y}`).join(' ')}
              style={{ fill: `rgba(0, 0,0, ${intersections * 0.1})` }}
            />
          ))}
      </svg>
    );
  }

  getIntersectionCount(poly) {
    var tPoly = turf.polygon([
      poly
        .map(e => [e[0].x, e[0].y])
        .concat([[_.first(poly)[0].x, _.first(poly)[0].y]])
    ]);
    return _.filter(this.state.itemPolys, itemPly =>
      turf.intersect(tPoly, itemPly)
    ).length;
  }
}
