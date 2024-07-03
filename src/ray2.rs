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

use crate::{Point2, Point2d, Point2f, Scalar, Vector2, Vector2d, Vector2f};

/// Ray in 2D space.
#[derive(Clone, PartialEq, Debug)]
pub struct Ray2<S: Scalar> {
    pub origin: Point2<S>,
    pub direction: Vector2<S>,
}

/// Alias for `Ray2<f32>`.
pub type Ray2f = Ray2<f32>;

/// Alias for `Ray2<f64>`.
pub type Ray2d = Ray2<f64>;

#[inline]
pub fn ray2<S: Scalar>(origin: Point2<S>, direction: Vector2<S>) -> Ray2<S> {
    Ray2::new(origin, direction)
}

#[inline]
pub fn ray2f(origin: Point2f, direction: Vector2f) -> Ray2f {
    Ray2f::new(origin, direction)
}

#[inline]
pub fn ray2d(origin: Point2d, direction: Vector2d) -> Ray2d {
    Ray2d::new(origin, direction)
}

// ===== Ray2 ==================================================================================================================================================

impl<S: Scalar> Ray2<S> {
    /// Creates and returns a new `Ray2` with an origin point and direction vector.
    #[inline]
    pub fn new(origin: Point2<S>, direction: Vector2<S>) -> Ray2<S> {
        Ray2 { origin, direction }
    }

    /// Computes and returns a point at a distance along this ray.
    #[inline]
    pub fn at(&self, distance: S) -> Point2<S> {
        self.origin + self.direction * distance
    }
}
