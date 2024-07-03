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

use std::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::{Dimension3, Distance, dot, Length, max, min, MinMax, RelativeDistance, Scalar, Vector3};

/// Point in 3D space.
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Point3<S: Scalar> {
    pub x: S,
    pub y: S,
    pub z: S,
}

/// Alias for `Point3<f32>`.
pub type Point3f = Point3<f32>;

/// Alias for `Point3<f64>`.
pub type Point3d = Point3<f64>;

#[inline]
pub fn point3<S: Scalar>(x: S, y: S, z: S) -> Point3<S> {
    Point3::new(x, y, z)
}

#[inline]
pub fn point3f(x: f32, y: f32, z: f32) -> Point3f {
    Point3f::new(x, y, z)
}

#[inline]
pub fn point3d(x: f64, y: f64, z: f64) -> Point3d {
    Point3d::new(x, y, z)
}

// ===== Point3 ================================================================================================================================================

impl<S: Scalar> Point3<S> {
    /// Creates and returns a new `Point3` with x, y and z coordinates.
    #[inline]
    pub fn new(x: S, y: S, z: S) -> Point3<S> {
        Point3 { x, y, z }
    }

    /// Returns a `Point3` which represents the origin (x = 0, y = 0 and z = 0).
    #[inline]
    pub fn origin() -> Point3<S> {
        Point3::new(S::zero(), S::zero(), S::zero())
    }

    /// Returns the dimension with the smallest extent of this point.
    #[inline]
    pub fn min_dimension(self) -> Dimension3 {
        let Point3 { x, y, z } = self.abs();
        if x <= y && x <= z { Dimension3::X } else if y <= z { Dimension3::Y } else { Dimension3::Z }
    }

    /// Returns the dimension with the largest extent of this point.
    #[inline]
    pub fn max_dimension(self) -> Dimension3 {
        let Point3 { x, y, z } = self.abs();
        if x > y && x > z { Dimension3::X } else if y > z { Dimension3::Y } else { Dimension3::Z }
    }

    /// Returns the element-wise floor of this point.
    #[inline]
    pub fn floor(self) -> Point3<S> {
        Point3::new(self.x.floor(), self.y.floor(), self.z.floor())
    }

    /// Returns the element-wise ceiling of this point.
    #[inline]
    pub fn ceil(self) -> Point3<S> {
        Point3::new(self.x.ceil(), self.y.ceil(), self.z.ceil())
    }

    /// Returns the element-wise rounded value of this point.
    #[inline]
    pub fn round(self) -> Point3<S> {
        Point3::new(self.x.round(), self.y.round(), self.z.round())
    }

    /// Returns the element-wise truncated value of this point.
    #[inline]
    pub fn trunc(self) -> Point3<S> {
        Point3::new(self.x.trunc(), self.y.trunc(), self.z.trunc())
    }

    /// Returns the element-wise fractional value of this point.
    #[inline]
    pub fn fract(self) -> Point3<S> {
        Point3::new(self.x.fract(), self.y.fract(), self.z.fract())
    }

    /// Returns the element-wise absolute value of this point.
    #[inline]
    pub fn abs(self) -> Point3<S> {
        Point3::new(self.x.abs(), self.y.abs(), self.z.abs())
    }

    /// Returns a point with a permutation of the elements of this point.
    #[inline]
    pub fn permute(self, dim_x: Dimension3, dim_y: Dimension3, dim_z: Dimension3) -> Point3<S> {
        Point3::new(self[dim_x], self[dim_y], self[dim_z])
    }
}

impl<S: Scalar> Index<Dimension3> for Point3<S> {
    type Output = S;

    #[inline]
    fn index(&self, dim: Dimension3) -> &S {
        match dim {
            Dimension3::X => &self.x,
            Dimension3::Y => &self.y,
            Dimension3::Z => &self.z,
        }
    }
}

impl<S: Scalar> IndexMut<Dimension3> for Point3<S> {
    #[inline]
    fn index_mut(&mut self, dim: Dimension3) -> &mut S {
        match dim {
            Dimension3::X => &mut self.x,
            Dimension3::Y => &mut self.y,
            Dimension3::Z => &mut self.z,
        }
    }
}

impl<S: Scalar> MinMax for Point3<S> {
    /// Returns the element-wise minimum of two points.
    #[inline]
    fn min(self, p: Point3<S>) -> Point3<S> {
        Point3::new(min(self.x, p.x), min(self.y, p.y), min(self.z, p.z))
    }

    /// Returns the element-wise maximum of two points.
    #[inline]
    fn max(self, p: Point3<S>) -> Point3<S> {
        Point3::new(max(self.x, p.x), max(self.y, p.y), max(self.z, p.z))
    }
}

impl<S: Scalar> Distance for Point3<S> {
    type Output = S;

    /// Computes and returns the distance between this and another point.
    #[inline]
    fn distance(self, p: Point3<S>) -> S {
        (p - self).length()
    }
}

impl<S: Scalar> RelativeDistance for Point3<S> {
    /// Returns `true` if `p1` is closer to this point than `p2`.
    ///
    /// This is more computationally efficient than computing the distance between this point and the points `p1` and `p2` and comparing the distances,
    /// because square root operations that are needed for computing the distances are avoided.
    #[inline]
    fn is_closer_to(self, p1: Point3<S>, p2: Point3<S>) -> bool {
        let (v1, v2) = (p1 - self, p2 - self);
        dot(v1, v1) < dot(v2, v2)
    }

    /// Returns `true` if `p1` is farther from this point than `p2`.
    ///
    /// This is more computationally efficient than computing the distance between this point and the points `p1` and `p2` and comparing the distances,
    /// because square root operations that are needed for computing the distances are avoided.
    #[inline]
    fn is_farther_from(self, p1: Point3<S>, p2: Point3<S>) -> bool {
        let (v1, v2) = (p1 - self, p2 - self);
        dot(v1, v1) > dot(v2, v2)
    }

    /// Checks which of the points `p1` and `p2` is closest to this point and returns the closest one.
    ///
    /// This is more computationally efficient than computing the distance between this point and the points `p1` and `p2` and comparing the distances,
    /// because square root operations that are needed for computing the distances are avoided.
    #[inline]
    fn closest(self, p1: Point3<S>, p2: Point3<S>) -> Point3<S> {
        if self.is_closer_to(p1, p2) { p1 } else { p2 }
    }

    /// Checks which of the points `p1` and `p2` is farthest from this point and returns the farthest one.
    ///
    /// This is more computationally efficient than computing the distance between this point and the points `p1` and `p2` and comparing the distances,
    /// because square root operations that are needed for computing the distances are avoided.
    #[inline]
    fn farthest(self, p1: Point3<S>, p2: Point3<S>) -> Point3<S> {
        if self.is_farther_from(p1, p2) { p1 } else { p2 }
    }
}

impl<S: Scalar> Add<Vector3<S>> for Point3<S> {
    type Output = Point3<S>;

    #[inline]
    fn add(self, v: Vector3<S>) -> Point3<S> {
        Point3::new(self.x + v.x, self.y + v.y, self.z + v.z)
    }
}

impl<S: Scalar> AddAssign<Vector3<S>> for Point3<S> {
    #[inline]
    fn add_assign(&mut self, v: Vector3<S>) {
        self.x += v.x;
        self.y += v.y;
        self.z += v.z;
    }
}

impl<S: Scalar> Sub<Vector3<S>> for Point3<S> {
    type Output = Point3<S>;

    #[inline]
    fn sub(self, v: Vector3<S>) -> Point3<S> {
        Point3::new(self.x - v.x, self.y - v.y, self.z - v.z)
    }
}

impl<S: Scalar> SubAssign<Vector3<S>> for Point3<S> {
    #[inline]
    fn sub_assign(&mut self, v: Vector3<S>) {
        self.x -= v.x;
        self.y -= v.y;
        self.z -= v.z;
    }
}

impl<S: Scalar> Sub<Point3<S>> for Point3<S> {
    type Output = Vector3<S>;

    #[inline]
    fn sub(self, p: Point3<S>) -> Vector3<S> {
        Vector3::new(self.x - p.x, self.y - p.y, self.z - p.z)
    }
}

impl<S: Scalar> Neg for Point3<S> {
    type Output = Point3<S>;

    #[inline]
    fn neg(self) -> Point3<S> {
        Point3::new(-self.x, -self.y, -self.z)
    }
}

impl<S: Scalar> Mul<S> for Point3<S> {
    type Output = Point3<S>;

    #[inline]
    fn mul(self, s: S) -> Point3<S> {
        Point3::new(self.x * s, self.y * s, self.z * s)
    }
}

impl<S: Scalar> MulAssign<S> for Point3<S> {
    #[inline]
    fn mul_assign(&mut self, s: S) {
        self.x *= s;
        self.y *= s;
        self.z *= s;
    }
}

impl Mul<Point3f> for f32 {
    type Output = Point3f;

    #[inline]
    fn mul(self, p: Point3f) -> Point3f {
        p * self
    }
}

impl Mul<Point3d> for f64 {
    type Output = Point3d;

    #[inline]
    fn mul(self, p: Point3d) -> Point3d {
        p * self
    }
}

impl<S: Scalar> Div<S> for Point3<S> {
    type Output = Point3<S>;

    #[inline]
    fn div(self, s: S) -> Point3<S> {
        Point3::new(self.x / s, self.y / s, self.z / s)
    }
}

impl<S: Scalar> DivAssign<S> for Point3<S> {
    #[inline]
    fn div_assign(&mut self, s: S) {
        self.x /= s;
        self.y /= s;
        self.z /= s;
    }
}

impl<S: Scalar> From<Vector3<S>> for Point3<S> {
    #[inline]
    fn from(v: Vector3<S>) -> Point3<S> {
        Point3::new(v.x, v.y, v.z)
    }
}
