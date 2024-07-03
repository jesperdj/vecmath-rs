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

use crate::{Point3, Point3d, Point3f, Scalar, Vector3, Vector3d, Vector3f};

/// Ray in 3D space.
#[derive(Clone, PartialEq, Debug)]
pub struct Ray3<S: Scalar> {
    pub origin: Point3<S>,
    pub direction: Vector3<S>,
}

/// Alias for `Ray3<f32>`.
pub type Ray3f = Ray3<f32>;

/// Alias for `Ray3<f64>`.
pub type Ray3d = Ray3<f64>;

#[inline]
pub fn ray3<S: Scalar>(origin: Point3<S>, direction: Vector3<S>) -> Ray3<S> {
    Ray3::new(origin, direction)
}

#[inline]
pub fn ray3f(origin: Point3f, direction: Vector3f) -> Ray3f {
    Ray3f::new(origin, direction)
}

#[inline]
pub fn ray3d(origin: Point3d, direction: Vector3d) -> Ray3d {
    Ray3d::new(origin, direction)
}

// ===== Ray3 ==================================================================================================================================================

impl<S: Scalar> Ray3<S> {
    /// Creates and returns a new `Ray3` with an origin point and direction vector.
    #[inline]
    pub fn new(origin: Point3<S>, direction: Vector3<S>) -> Ray3<S> {
        Ray3 { origin, direction }
    }

    /// Computes and returns a point at a distance along this ray.
    #[inline]
    pub fn at(&self, distance: S) -> Point3<S> {
        self.origin + self.direction * distance
    }
}
