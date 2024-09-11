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

use crate::{max, min, min_max, Dimension2, Dimension3, DotProduct, Point2, Point3, Ray2, Ray3, Scalar, Union, Vector2, Vector3};
use std::ops::Range;

/// Axis-aligned bounding box in 2D space.
#[derive(Clone, PartialEq, Debug)]
pub struct BoundingBox2<S: Scalar> {
    /// Minimum corner point.
    pub min: Point2<S>,

    /// Maximum corner point.
    pub max: Point2<S>,
}

/// Axis-aligned bounding box in 3D space.
#[derive(Clone, PartialEq, Debug)]
pub struct BoundingBox3<S: Scalar> {
    /// Minimum corner point.
    pub min: Point3<S>,

    /// Maximum corner point.
    pub max: Point3<S>,
}

// ===== BoundingBox2 ==========================================================================================================================================

impl<S: Scalar> BoundingBox2<S> {
    /// Creates and returns a new bounding box with minimum and maximum corner points.
    pub fn new(p1: Point2<S>, p2: Point2<S>) -> BoundingBox2<S> {
        let (min, max) = min_max(p1, p2);

        BoundingBox2 { min, max }
    }

    /// Returns an empty bounding box.
    pub fn empty() -> BoundingBox2<S> {
        BoundingBox2 {
            min: Point2 { x: S::infinity(), y: S::infinity() },
            max: Point2 { x: S::neg_infinity(), y: S::neg_infinity() },
        }
    }

    /// Returns an infinite bounding box which contains all of 2D space.
    pub fn infinite() -> BoundingBox2<S> {
        BoundingBox2 {
            min: Point2 { x: S::neg_infinity(), y: S::neg_infinity() },
            max: Point2 { x: S::infinity(), y: S::infinity() },
        }
    }

    /// Creates a bounding box from a corner point and a diagonal vector.
    pub fn from_corner_and_diagonal(corner: Point2<S>, diagonal: Vector2<S>) -> BoundingBox2<S> {
        let (mut min_corner, mut max_corner) = (corner, corner);
        for i in 1..=3 {
            let mut c = corner;
            if i & 0b01 != 0 { c.x += diagonal.x; }
            if i & 0b10 != 0 { c.y += diagonal.y; }

            min_corner = min(min_corner, c);
            max_corner = max(max_corner, c);
        }

        BoundingBox2 { min: min_corner, max: max_corner }
    }

    /// Returns the width (extent in the X dimension) of this bounding box.
    pub fn width(&self) -> S {
        self.max.x - self.min.x
    }

    /// Returns the height (extent in the Y dimension) of this bounding box.
    pub fn height(&self) -> S {
        self.max.y - self.min.y
    }

    /// Returns the size (extent in X and Y dimensions) of this bounding box.
    pub fn size(&self) -> (S, S) {
        (self.width(), self.height())
    }

    /// Returns the extent of this bounding box in a dimension.
    pub fn extent(&self, dim: Dimension2) -> S {
        match dim {
            Dimension2::X => self.width(),
            Dimension2::Y => self.height(),
        }
    }

    /// Returns the dimension with the smallest extent of this bounding box.
    pub fn min_dimension(&self) -> Dimension2 {
        let d = self.diagonal();
        if d.x <= d.y { Dimension2::X } else { Dimension2::Y }
    }

    /// Returns the dimension with the largest extent of this bounding box.
    pub fn max_dimension(&self) -> Dimension2 {
        let d = self.diagonal();
        if d.x > d.y { Dimension2::X } else { Dimension2::Y }
    }

    /// Returns the area (width times height) of this bounding box.
    pub fn area(&self) -> S {
        let (width, height) = self.size();
        width * height
    }

    /// Returns the center point of this bounding box.
    pub fn center(&self) -> Point2<S> {
        self.min + self.diagonal() * S::from(0.5).unwrap()
    }

    /// Returns a corner point of this bounding box, indicated by an index (which must be in the range `0..=3`).
    pub fn corner(&self, index: u32) -> Point2<S> {
        debug_assert!(index < 4, "Invalid corner index: {}", index);

        Point2 {
            x: if index & 0b01 == 0 { self.min.x } else { self.max.x },
            y: if index & 0b10 == 0 { self.min.y } else { self.max.y },
        }
    }

    /// Returns the diagonal of this bounding box as a vector.
    pub fn diagonal(&self) -> Vector2<S> {
        self.max - self.min
    }

    /// Checks if two bounding boxes overlap.
    pub fn overlaps(&self, bb: &BoundingBox2<S>) -> bool {
        self.max.x >= bb.min.x && self.min.x <= bb.max.x
            && self.max.y >= bb.min.y && self.min.y <= bb.max.y
    }

    /// Checks if a point is inside this bounding box.
    pub fn is_inside(&self, p: Point2<S>) -> bool {
        p.x >= self.min.x && p.x <= self.max.x
            && p.y >= self.min.y && p.y <= self.max.y
    }

    /// Computes and returns the intersection between two bounding boxes.
    ///
    /// The intersection is the largest bounding box that contains the region where the two bounding boxes overlap.
    ///
    /// Returns `Some` when the bounding boxes overlap; `None` if the bounding boxes do not overlap.
    pub fn intersection(&self, bb: &BoundingBox2<S>) -> Option<BoundingBox2<S>> {
        if self.overlaps(bb) {
            Some(BoundingBox2 {
                min: max(self.min, bb.min),
                max: min(self.max, bb.max),
            })
        } else {
            None
        }
    }

    /// Computes the intersections of this bounding box with a ray.
    ///
    /// Returns a `Some` containing the range in which the ray intersects the bounding box, or `None` if the ray does not intersect the bounding box.
    pub fn intersect_ray(&self, ray: Ray2<S>) -> Option<Range<S>> {
        let (start, end) = (ray.range.start, ray.range.end);

        let d1 = (self.min.x - ray.origin.x) / ray.direction.x;
        let d2 = (self.max.x - ray.origin.x) / ray.direction.x;

        let start = max(start, min(d1, d2));
        let end = min(end, max(d1, d2));

        if start > end { return None; }

        let d1 = (self.min.y - ray.origin.y) / ray.direction.y;
        let d2 = (self.max.y - ray.origin.y) / ray.direction.y;

        let start = max(start, min(d1, d2));
        let end = min(end, max(d1, d2));

        if start < end { Some(start..end) } else { None }
    }
}

impl<S: Scalar> Union for &BoundingBox2<S> {
    type Output = BoundingBox2<S>;

    /// Computes and returns the union between two bounding boxes.
    ///
    /// The union is the smallest bounding box that contains both bounding boxes.
    fn union(self, bb: &BoundingBox2<S>) -> BoundingBox2<S> {
        BoundingBox2 {
            min: min(self.min, bb.min),
            max: max(self.max, bb.max),
        }
    }
}

impl<S: Scalar> Union<Point2<S>> for &BoundingBox2<S> {
    type Output = BoundingBox2<S>;

    /// Computes and returns the union between this bounding box and a point.
    ///
    /// The union is the smallest bounding box that contains both the bounding box and the point.
    fn union(self, p: Point2<S>) -> BoundingBox2<S> {
        BoundingBox2 {
            min: min(self.min, p),
            max: max(self.max, p),
        }
    }
}

// ===== BoundingBox3 ==========================================================================================================================================

impl<S: Scalar> BoundingBox3<S> {
    /// Creates and returns a new bounding box with minimum and maximum corner points.
    pub fn new(p1: Point3<S>, p2: Point3<S>) -> BoundingBox3<S> {
        let (min, max) = min_max(p1, p2);

        BoundingBox3 { min, max }
    }

    /// Returns an empty bounding box.
    pub fn empty() -> BoundingBox3<S> {
        BoundingBox3 {
            min: Point3 { x: S::infinity(), y: S::infinity(), z: S::infinity() },
            max: Point3 { x: S::neg_infinity(), y: S::neg_infinity(), z: S::neg_infinity() },
        }
    }

    /// Returns an infinite bounding box which contains all of 3D space.
    pub fn infinite() -> BoundingBox3<S> {
        BoundingBox3 {
            min: Point3 { x: S::neg_infinity(), y: S::neg_infinity(), z: S::neg_infinity() },
            max: Point3 { x: S::infinity(), y: S::infinity(), z: S::infinity() },
        }
    }

    /// Creates a bounding box from a corner point and a diagonal vector.
    pub fn from_corner_and_diagonal(corner: Point3<S>, diagonal: Vector3<S>) -> BoundingBox3<S> {
        let (mut min_corner, mut max_corner) = (corner, corner);
        for i in 1..=7 {
            let mut c = corner;
            if i & 0b001 != 0 { c.x += diagonal.x; }
            if i & 0b010 != 0 { c.y += diagonal.y; }
            if i & 0b100 != 0 { c.z += diagonal.z; }

            min_corner = min(min_corner, c);
            max_corner = max(max_corner, c);
        }

        BoundingBox3 { min: min_corner, max: max_corner }
    }

    /// Returns the width (extent in the X dimension) of this bounding box.
    pub fn width(&self) -> S {
        self.max.x - self.min.x
    }

    /// Returns the height (extent in the Y dimension) of this bounding box.
    pub fn height(&self) -> S {
        self.max.y - self.min.y
    }

    /// Returns the depth (extent in the Z dimension) of this bounding box.
    pub fn depth(&self) -> S {
        self.max.z - self.min.z
    }

    /// Returns the size (extent in X, Y and Z dimensions) of this bounding box.
    pub fn size(&self) -> (S, S, S) {
        (self.width(), self.height(), self.depth())
    }

    /// Returns the extent of this bounding box in a dimension.
    pub fn extent(&self, dim: Dimension3) -> S {
        match dim {
            Dimension3::X => self.width(),
            Dimension3::Y => self.height(),
            Dimension3::Z => self.depth(),
        }
    }

    /// Returns the dimension with the smallest extent of this bounding box.
    pub fn min_dimension(&self) -> Dimension3 {
        let d = self.diagonal();
        if d.x <= d.y && d.x <= d.z { Dimension3::X } else if d.y <= d.z { Dimension3::Y } else { Dimension3::Z }
    }

    /// Returns the dimension with the largest extent of this bounding box.
    pub fn max_dimension(&self) -> Dimension3 {
        let d = self.diagonal();
        if d.x > d.y && d.x > d.z { Dimension3::X } else if d.y > d.z { Dimension3::Y } else { Dimension3::Z }
    }

    /// Returns the surface area of this bounding box.
    pub fn surface_area(&self) -> S {
        let d = self.diagonal();
        S::from(2.0).unwrap() * d.dot(d)
    }

    /// Returns the volume (width times height times depth) of this bounding box.
    pub fn volume(&self) -> S {
        let (width, height, depth) = self.size();
        width * height * depth
    }

    /// Returns the center point of this bounding box.
    pub fn center(&self) -> Point3<S> {
        self.min + self.diagonal() * S::from(0.5).unwrap()
    }

    /// Returns a corner point of this bounding box, indicated by an index (which must be in the range `0..=7`).
    pub fn corner(&self, index: u32) -> Point3<S> {
        debug_assert!(index < 8, "Invalid corner index: {}", index);

        Point3 {
            x: if index & 0b001 == 0 { self.min.x } else { self.max.x },
            y: if index & 0b010 == 0 { self.min.y } else { self.max.y },
            z: if index & 0b100 == 0 { self.min.z } else { self.max.z },
        }
    }

    /// Returns the diagonal of this bounding box as a vector.
    pub fn diagonal(&self) -> Vector3<S> {
        self.max - self.min
    }

    /// Checks if two bounding boxes overlap.
    pub fn overlaps(&self, bb: &BoundingBox3<S>) -> bool {
        self.max.x >= bb.min.x && self.min.x <= bb.max.x
            && self.max.y >= bb.min.y && self.min.y <= bb.max.y
            && self.max.z >= bb.min.z && self.min.z <= bb.max.z
    }

    /// Checks if a point is inside this bounding box.
    pub fn is_inside(&self, p: Point3<S>) -> bool {
        p.x >= self.min.x && p.x <= self.max.x
            && p.y >= self.min.y && p.y <= self.max.y
            && p.z >= self.min.z && p.z <= self.max.z
    }

    /// Computes and returns the intersection between two bounding boxes.
    ///
    /// The intersection is the largest bounding box that contains the region where the two bounding boxes overlap.
    ///
    /// Returns `Some` when the bounding boxes overlap; `None` if the bounding boxes do not overlap.
    pub fn intersection(&self, bb: &BoundingBox3<S>) -> Option<BoundingBox3<S>> {
        if self.overlaps(bb) {
            Some(BoundingBox3 {
                min: max(self.min, bb.min),
                max: min(self.max, bb.max),
            })
        } else {
            None
        }
    }

    /// Computes the intersections of this bounding box with a ray.
    ///
    /// Returns a `Some` containing the range in which the ray intersects the bounding box, or `None` if the ray does not intersect the bounding box.
    pub fn intersect_ray(&self, ray: Ray3<S>) -> Option<Range<S>> {
        let (start, end) = (ray.range.start, ray.range.end);

        let d1 = (self.min.x - ray.origin.x) / ray.direction.x;
        let d2 = (self.max.x - ray.origin.x) / ray.direction.x;

        let start = max(start, min(d1, d2));
        let end = min(end, max(d1, d2));

        if start > end { return None; }

        let d1 = (self.min.y - ray.origin.y) / ray.direction.y;
        let d2 = (self.max.y - ray.origin.y) / ray.direction.y;

        let start = max(start, min(d1, d2));
        let end = min(end, max(d1, d2));

        if start > end { return None; }

        let d1 = (self.min.z - ray.origin.z) / ray.direction.z;
        let d2 = (self.max.z - ray.origin.z) / ray.direction.z;

        let start = max(start, min(d1, d2));
        let end = min(end, max(d1, d2));

        if start < end { Some(start..end) } else { None }
    }
}

impl<S: Scalar> Union<&BoundingBox3<S>> for &BoundingBox3<S> {
    type Output = BoundingBox3<S>;

    /// Computes and returns the union between two bounding boxes.
    ///
    /// The union is the smallest bounding box that contains both bounding boxes.
    fn union(self, bb: &BoundingBox3<S>) -> BoundingBox3<S> {
        BoundingBox3 {
            min: min(self.min, bb.min),
            max: max(self.max, bb.max),
        }
    }
}

impl<S: Scalar> Union<Point3<S>> for &BoundingBox3<S> {
    type Output = BoundingBox3<S>;

    /// Computes and returns the union between this bounding box and a point.
    ///
    /// The union is the smallest bounding box that contains both the bounding box and the point.
    fn union(self, p: Point3<S>) -> BoundingBox3<S> {
        BoundingBox3 {
            min: min(self.min, p),
            max: max(self.max, p),
        }
    }
}
