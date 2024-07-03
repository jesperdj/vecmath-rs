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

use crate::{Dimension3, Intersection, max, min, Point3, Ray3, Scalar, Union, Vector3};

/// Axis-aligned bounding box in 3D space.
#[derive(Clone, PartialEq, Debug)]
pub struct BoundingBox3<S: Scalar> {
    pub min: Point3<S>,
    pub max: Point3<S>,
}

/// Alias for `BoundingBox3<f32>`.
pub type BoundingBox3f = BoundingBox3<f32>;

/// Alias for `BoundingBox2<f64>`.
pub type BoundingBox3d = BoundingBox3<f64>;

// ===== BoundingBox3 ==========================================================================================================================================

impl<S: Scalar> BoundingBox3<S> {
    /// Creates and returns a new `BoundingBox3` with minimum and maximum corner points.
    #[inline]
    pub fn new(p1: Point3<S>, p2: Point3<S>) -> BoundingBox3<S> {
        let min = min(p1, p2);
        let max = max(p1, p2);

        BoundingBox3 { min, max }
    }

    /// Returns an empty `BoundingBox3`.
    #[inline]
    pub fn empty() -> BoundingBox3<S> {
        BoundingBox3 {
            min: Point3::new(S::infinity(), S::infinity(), S::infinity()),
            max: Point3::new(S::neg_infinity(), S::neg_infinity(), S::neg_infinity()),
        }
    }

    /// Returns an infinite `BoundingBox3` which contains all of 3D space.
    #[inline]
    pub fn infinite() -> BoundingBox3<S> {
        BoundingBox3 {
            min: Point3::new(S::neg_infinity(), S::neg_infinity(), S::neg_infinity()),
            max: Point3::new(S::infinity(), S::infinity(), S::infinity()),
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

    /// Returns the depth (extent in the Z dimension) of this bounding box.
    #[inline]
    pub fn depth(&self) -> S {
        self.max.z - self.min.z
    }

    /// Returns the size (extent in X, Y and Z dimensions) of this bounding box.
    #[inline]
    pub fn size(&self) -> (S, S, S) {
        (self.width(), self.height(), self.depth())
    }

    /// Returns the extent of this bounding box in a dimension.
    #[inline]
    pub fn extent(&self, dim: Dimension3) -> S {
        match dim {
            Dimension3::X => self.width(),
            Dimension3::Y => self.height(),
            Dimension3::Z => self.depth(),
        }
    }

    /// Returns the dimension with the smallest extent of this bounding box.
    #[inline]
    pub fn min_dimension(&self) -> Dimension3 {
        let d = self.diagonal();
        if d.x <= d.y && d.x <= d.z { Dimension3::X } else if d.y <= d.z { Dimension3::Y } else { Dimension3::Z }
    }

    /// Returns the dimension with the largest extent of this bounding box.
    #[inline]
    pub fn max_dimension(&self) -> Dimension3 {
        let d = self.diagonal();
        if d.x > d.y && d.x > d.z { Dimension3::X } else if d.y > d.z { Dimension3::Y } else { Dimension3::Z }
    }

    /// Returns the surface area of this bounding box.
    #[inline]
    pub fn surface_area(&self) -> S {
        let d = self.diagonal();
        S::two() * (d.x * d.y + d.x * d.z + d.y * d.z)
    }

    /// Returns the volume (width times height times depth) of this bounding box.
    #[inline]
    pub fn volume(&self) -> S {
        let (width, height, depth) = self.size();
        width * height * depth
    }

    /// Returns the center point of this bounding box.
    #[inline]
    pub fn center(&self) -> Point3<S> {
        self.min + self.diagonal() * S::half()
    }

    /// Returns a corner point of this bounding box, indicated by an index (which must be in the range `0..=7`).
    #[inline]
    pub fn corner(&self, index: u32) -> Point3<S> {
        debug_assert!(index < 8, "Invalid corner index: {}", index);
        let x = if index & 0b001 == 0 { self.min.x } else { self.max.x };
        let y = if index & 0b010 == 0 { self.min.y } else { self.max.y };
        let z = if index & 0b100 == 0 { self.min.z } else { self.max.z };
        Point3::new(x, y, z)
    }

    /// Returns the diagonal of this bounding box as a vector.
    #[inline]
    pub fn diagonal(&self) -> Vector3<S> {
        self.max - self.min
    }

    /// Checks if two bounding boxes overlap.
    #[inline]
    pub fn overlaps(&self, bb: &BoundingBox3<S>) -> bool {
        //@formatter:off
        self.max.x >= bb.min.x && self.min.x <= bb.max.x &&
        self.max.y >= bb.min.y && self.min.y <= bb.max.y &&
        self.max.z >= bb.min.z && self.min.z <= bb.max.z
        //@formatter:on
    }

    /// Checks if a point is inside this bounding box.
    #[inline]
    pub fn is_inside(&self, p: Point3<S>) -> bool {
        //@formatter:off
        p.x >= self.min.x && p.x <= self.max.x &&
        p.y >= self.min.y && p.y <= self.max.y &&
        p.z >= self.min.z && p.z <= self.max.z
        //@formatter:on
    }
}

impl<S: Scalar> Union<&BoundingBox3<S>> for &BoundingBox3<S> {
    type Output = BoundingBox3<S>;

    /// Computes and returns the union between two bounding boxes.
    ///
    /// The union is the smallest bounding box that contains both bounding boxes.
    #[inline]
    fn union(self, bb: &BoundingBox3<S>) -> BoundingBox3<S> {
        BoundingBox3::new(min(self.min, bb.min), max(self.max, bb.max))
    }
}

impl<S: Scalar> Union<Point3<S>> for &BoundingBox3<S> {
    type Output = BoundingBox3<S>;

    /// Computes and returns the union between this bounding box and a point.
    ///
    /// The union is the smallest bounding box that contains both the bounding box and the point.
    #[inline]
    fn union(self, p: Point3<S>) -> BoundingBox3<S> {
        BoundingBox3::new(min(self.min, p), max(self.max, p))
    }
}

impl<S: Scalar> Intersection<&BoundingBox3<S>> for &BoundingBox3<S> {
    type Output = BoundingBox3<S>;

    /// Computes and returns the intersection between two bounding boxes.
    ///
    /// The intersection is the largest bounding box that contains the region where the two bounding boxes overlap.
    ///
    /// Returns `Some` when the bounding boxes overlap; `None` if the bounding boxes do not overlap.
    #[inline]
    fn intersect(self, bb: &BoundingBox3<S>) -> Option<BoundingBox3<S>> {
        if self.overlaps(bb) {
            Some(BoundingBox3::new(max(self.min, bb.min), min(self.max, bb.max)))
        } else {
            None
        }
    }
}

impl<S: Scalar> Intersection<&Ray3<S>> for &BoundingBox3<S> {
    type Output = Range<S>;

    /// Computes the intersections of this bounding box with a ray.
    ///
    /// Returns a `Some` containing the range in which the ray intersects the bounding box, or `None` if the ray does not intersect the bounding box.
    fn intersect(self, ray: &Ray3<S>) -> Option<Range<S>> {
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

        if start > end {
            return None;
        }

        let d1 = (self.min.z - ray.origin.z) / ray.direction.z;
        let d2 = (self.max.z - ray.origin.z) / ray.direction.z;

        let start = max(start, min(d1, d2));
        let end = min(end, max(d1, d2));

        if start <= end {
            Some(start..end)
        } else {
            None
        }
    }
}
