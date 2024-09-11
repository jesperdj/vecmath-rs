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

use crate::{Point2, Point3, Scalar, Vector2, Vector3};
use std::ops::Range;

/// Ray in 2D space.
///
/// A ray has an origin (a [point](crate::Point2)), a direction (a [vector](crate::Vector2)) and a range of valid distance values.
#[derive(Clone, PartialEq, Debug)]
pub struct Ray2<S: Scalar> {
    pub origin: Point2<S>,
    pub direction: Vector2<S>,
    pub range: Range<S>,
}

/// Ray in 3D space.
///
/// A ray has an origin (a [point](crate::Point3)), a direction (a [vector](crate::Vector3)) and a range of valid distance values.
#[derive(Clone, PartialEq, Debug)]
pub struct Ray3<S: Scalar> {
    pub origin: Point3<S>,
    pub direction: Vector3<S>,
    pub range: Range<S>,
}

// ===== Ray2 ==================================================================================================================================================

impl<S: Scalar> Ray2<S> {
    /// Creates and returns a new ray.
    pub fn new(origin: Point2<S>, direction: Vector2<S>, range: Range<S>) -> Ray2<S> {
        Ray2 { origin, direction, range }
    }

    /// Returns the point that is at the specified distance of the origin of the ray, along the direction of the ray.
    ///
    /// Returns `None` if the distance is not in the range of the ray.
    pub fn at(&self, distance: S) -> Option<Point2<S>> {
        if self.is_in_range(distance) { Some(self.origin + self.direction * distance) } else { None }
    }

    /// Returns `true` if the distance is in the range of the ray, `false` otherwise.
    pub fn is_in_range(&self, distance: S) -> bool {
        self.range.contains(&distance)
    }
}

// ===== Ray3 ==================================================================================================================================================

impl<S: Scalar> Ray3<S> {
    /// Creates and returns a new ray.
    pub fn new(origin: Point3<S>, direction: Vector3<S>, range: Range<S>) -> Ray3<S> {
        Ray3 { origin, direction, range }
    }

    /// Returns the point that is at the specified distance of the origin of the ray, along the direction of the ray.
    ///
    /// Returns `None` if the distance is not in the range of the ray.
    pub fn at(&self, distance: S) -> Option<Point3<S>> {
        if self.is_in_range(distance) { Some(self.origin + self.direction * distance) } else { None }
    }

    /// Returns `true` if the distance is in the range of the ray, `false` otherwise.
    pub fn is_in_range(&self, distance: S) -> bool {
        self.range.contains(&distance)
    }
}
