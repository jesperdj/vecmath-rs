// Copyright 2024 Jesper de Jong
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::ops::Range;

use crate::{Dimension2, Intersection, max, min, Point2, Ray2, Scalar, Union, Vector2};

/// Axis-aligned bounding box in 2D space.
#[derive(Clone, PartialEq, Debug)]
pub struct BoundingBox2<S: Scalar> {
    pub min: Point2<S>,
    pub max: Point2<S>,
}

/// Alias for `BoundingBox2<f32>`.
pub type BoundingBox2f = BoundingBox2<f32>;

/// Alias for `BoundingBox2<f64>`.
pub type BoundingBox2d = BoundingBox2<f64>;

// ===== BoundingBox2 ==========================================================================================================================================

impl<S: Scalar> BoundingBox2<S> {
    /// Creates and returns a new `BoundingBox2` with minimum and maximum corner points.
    #[inline]
    pub fn new(p1: Point2<S>, p2: Point2<S>) -> BoundingBox2<S> {
        let min = min(p1, p2);
        let max = max(p1, p2);

        BoundingBox2 { min, max }
    }

    /// Returns an empty `BoundingBox2`.
    #[inline]
    pub fn empty() -> BoundingBox2<S> {
        BoundingBox2 {
            min: Point2::new(S::infinity(), S::infinity()),
            max: Point2::new(S::neg_infinity(), S::neg_infinity()),
        }
    }

    /// Returns an infinite `BoundingBox2` which contains all of 2D space.
    #[inline]
    pub fn infinite() -> BoundingBox2<S> {
        BoundingBox2 {
            min: Point2::new(S::neg_infinity(), S::neg_infinity()),
            max: Point2::new(S::infinity(), S::infinity()),
        }
    }

    /// Returns the width (extent in the X dimension) of this bounding box.
    #[inline]
    pub fn width(&self) -> S {
        self.max.x - self.min.x
    }

    /// Returns the height (extent in the Y dimension) of this bounding box.
    #[inline]
    pub fn height(&self) -> S {
        self.max.y - self.min.y
    }

    /// Returns the size (extent in X and Y dimensions) of this bounding box.
    #[inline]
    pub fn size(&self) -> (S, S) {
        (self.width(), self.height())
    }

    /// Returns the extent of this bounding box in a dimension.
    #[inline]
    pub fn extent(&self, dim: Dimension2) -> S {
        match dim {
            Dimension2::X => self.width(),
            Dimension2::Y => self.height(),
        }
    }

    /// Returns the dimension with the smallest extent of this bounding box.
    #[inline]
    pub fn min_dimension(&self) -> Dimension2 {
        let d = self.diagonal();
        if d.x <= d.y { Dimension2::X } else { Dimension2::Y }
    }

    /// Returns the dimension with the largest extent of this bounding box.
    #[inline]
    pub fn max_dimension(&self) -> Dimension2 {
        let d = self.diagonal();
        if d.x > d.y { Dimension2::X } else { Dimension2::Y }
    }

    /// Returns the area (width times height) of this bounding box.
    #[inline]
    pub fn area(&self) -> S {
        let (width, height) = self.size();
        width * height
    }

    /// Returns the center point of this bounding box.
    #[inline]
    pub fn center(&self) -> Point2<S> {
        self.min + self.diagonal() * S::half()
    }

    /// Returns a corner point of this bounding box, indicated by an index (which must be in the range `0..=3`).
    #[inline]
    pub fn corner(&self, index: u32) -> Point2<S> {
        debug_assert!(index < 4, "Invalid corner index: {}", index);
        let x = if index & 0b01 == 0 { self.min.x } else { self.max.x };
        let y = if index & 0b10 == 0 { self.min.y } else { self.max.y };
        Point2::new(x, y)
    }

    /// Returns the diagonal of this bounding box as a vector.
    #[inline]
    pub fn diagonal(&self) -> Vector2<S> {
        self.max - self.min
    }

    /// Checks if two bounding boxes overlap.
    #[inline]
    pub fn overlaps(&self, bb: &BoundingBox2<S>) -> bool {
        //@formatter:off
        self.max.x >= bb.min.x && self.min.x <= bb.max.x &&
        self.max.y >= bb.min.y && self.min.y <= bb.max.y
        //@formatter:on
    }

    /// Checks if a point is inside this bounding box.
    #[inline]
    pub fn is_inside(&self, p: Point2<S>) -> bool {
        //@formatter:off
        p.x >= self.min.x && p.x <= self.max.x &&
        p.y >= self.min.y && p.y <= self.max.y
        //@formatter:on
    }
}

impl<S: Scalar> Union<&BoundingBox2<S>> for &BoundingBox2<S> {
    type Output = BoundingBox2<S>;

    /// Computes and returns the union between two bounding boxes.
    ///
    /// The union is the smallest bounding box that contains both bounding boxes.
    #[inline]
    fn union(self, bb: &BoundingBox2<S>) -> BoundingBox2<S> {
        BoundingBox2::new(min(self.min, bb.min), max(self.max, bb.max))
    }
}

impl<S: Scalar> Union<Point2<S>> for &BoundingBox2<S> {
    type Output = BoundingBox2<S>;

    /// Computes and returns the union between this bounding box and a point.
    ///
    /// The union is the smallest bounding box that contains both the bounding box and the point.
    #[inline]
    fn union(self, p: Point2<S>) -> BoundingBox2<S> {
        BoundingBox2::new(min(self.min, p), max(self.max, p))
    }
}

impl<S: Scalar> Intersection<&BoundingBox2<S>> for &BoundingBox2<S> {
    type Output = BoundingBox2<S>;

    /// Computes and returns the intersection between two bounding boxes.
    ///
    /// The intersection is the largest bounding box that contains the region where the two bounding boxes overlap.
    ///
    /// Returns `Some` when the bounding boxes overlap; `None` if the bounding boxes do not overlap.
    #[inline]
    fn intersect(self, bb: &BoundingBox2<S>) -> Option<BoundingBox2<S>> {
        if self.overlaps(bb) {
            Some(BoundingBox2::new(max(self.min, bb.min), min(self.max, bb.max)))
        } else {
            None
        }
    }
}

impl<S: Scalar> Intersection<&Ray2<S>> for &BoundingBox2<S> {
    type Output = Range<S>;

    /// Computes the intersections of this bounding box with a ray.
    ///
    /// Returns a `Some` containing the range in which the ray intersects the bounding box, or `None` if the ray does not intersect the bounding box.
    fn intersect(self, ray: &Ray2<S>) -> Option<Range<S>> {
        let (start, end) = (S::zero(), S::infinity());

        let d1 = (self.min.x - ray.origin.x) / ray.direction.x;
        let d2 = (self.max.x - ray.origin.x) / ray.direction.x;

        let start = max(start, min(d1, d2));
        let end = min(end, max(d1, d2));

        if start > end {
            return None;
        }

        let d1 = (self.min.y - ray.origin.y) / ray.direction.y;
        let d2 = (self.max.y - ray.origin.y) / ray.direction.y;

        let start = max(start, min(d1, d2));
        let end = min(end, max(d1, d2));

        if start <= end {
            Some(start..end)
        } else {
            None
        }
    }
}
