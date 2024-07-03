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

use crate::{Dimension2, Distance, dot, Length, max, min, MinMax, RelativeDistance, Scalar, Vector2};

/// Point in 2D space.
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Point2<S: Scalar> {
    pub x: S,
    pub y: S,
}

/// Alias for `Point2<f32>`.
pub type Point2f = Point2<f32>;

/// Alias for `Point2<f64>`.
pub type Point2d = Point2<f64>;

#[inline]
pub fn point2<S: Scalar>(x: S, y: S) -> Point2<S> {
    Point2::new(x, y)
}

#[inline]
pub fn point2f(x: f32, y: f32) -> Point2f {
    Point2f::new(x, y)
}

#[inline]
pub fn point2d(x: f64, y: f64) -> Point2d {
    Point2d::new(x, y)
}

// ===== Point2 ================================================================================================================================================

impl<S: Scalar> Point2<S> {
    /// Creates and returns a new `Point2` with x and y coordinates.
    #[inline]
    pub fn new(x: S, y: S) -> Point2<S> {
        Point2 { x, y }
    }

    /// Returns a `Point2` which represents the origin (x = 0 and y = 0).
    #[inline]
    pub fn origin() -> Point2<S> {
        Point2::new(S::zero(), S::zero())
    }

    /// Returns the dimension with the smallest extent of this point.
    #[inline]
    pub fn min_dimension(self) -> Dimension2 {
        let Point2 { x, y } = self.abs();
        if x <= y { Dimension2::X } else { Dimension2::Y }
    }

    /// Returns the dimension with the largest extent of this point.
    #[inline]
    pub fn max_dimension(self) -> Dimension2 {
        let Point2 { x, y } = self.abs();
        if x > y { Dimension2::X } else { Dimension2::Y }
    }

    /// Returns the element-wise floor of this point.
    #[inline]
    pub fn floor(self) -> Point2<S> {
        Point2::new(self.x.floor(), self.y.floor())
    }

    /// Returns the element-wise ceiling of this point.
    #[inline]
    pub fn ceil(self) -> Point2<S> {
        Point2::new(self.x.ceil(), self.y.ceil())
    }

    /// Returns the element-wise rounded value of this point.
    #[inline]
    pub fn round(self) -> Point2<S> {
        Point2::new(self.x.round(), self.y.round())
    }

    /// Returns the element-wise truncated value of this point.
    #[inline]
    pub fn trunc(self) -> Point2<S> {
        Point2::new(self.x.trunc(), self.y.trunc())
    }

    /// Returns the element-wise fractional value of this point.
    #[inline]
    pub fn fract(self) -> Point2<S> {
        Point2::new(self.x.fract(), self.y.fract())
    }

    /// Returns the element-wise absolute value of this point.
    #[inline]
    pub fn abs(self) -> Point2<S> {
        Point2::new(self.x.abs(), self.y.abs())
    }

    /// Returns a point with a permutation of the elements of this point.
    #[inline]
    pub fn permute(self, dim_x: Dimension2, dim_y: Dimension2) -> Point2<S> {
        Point2::new(self[dim_x], self[dim_y])
    }
}

impl<S: Scalar> Index<Dimension2> for Point2<S> {
    type Output = S;

    #[inline]
    fn index(&self, dim: Dimension2) -> &S {
        match dim {
            Dimension2::X => &self.x,
            Dimension2::Y => &self.y,
        }
    }
}

impl<S: Scalar> IndexMut<Dimension2> for Point2<S> {
    #[inline]
    fn index_mut(&mut self, dim: Dimension2) -> &mut S {
        match dim {
            Dimension2::X => &mut self.x,
            Dimension2::Y => &mut self.y,
        }
    }
}

impl<S: Scalar> MinMax for Point2<S> {
    /// Returns the element-wise minimum of two points.
    #[inline]
    fn min(self, p: Point2<S>) -> Point2<S> {
        Point2::new(min(self.x, p.x), min(self.y, p.y))
    }

    /// Returns the element-wise maximum of two points.
    #[inline]
    fn max(self, p: Point2<S>) -> Point2<S> {
        Point2::new(max(self.x, p.x), max(self.y, p.y))
    }
}

impl<S: Scalar> Distance for Point2<S> {
    type Output = S;

    /// Computes and returns the distance between this and another point.
    #[inline]
    fn distance(self, p: Point2<S>) -> S {
        (p - self).length()
    }
}

impl<S: Scalar> RelativeDistance for Point2<S> {
    /// Returns `true` if `p1` is closer to this point than `p2`.
    ///
    /// This is more computationally efficient than computing the distance between this point and the points `p1` and `p2` and comparing the distances,
    /// because square root operations that are needed for computing the distances are avoided.
    #[inline]
    fn is_closer_to(self, p1: Point2<S>, p2: Point2<S>) -> bool {
        let (v1, v2) = (p1 - self, p2 - self);
        dot(v1, v1) < dot(v2, v2)
    }

    /// Returns `true` if `p1` is farther from this point than `p2`.
    ///
    /// This is more computationally efficient than computing the distance between this point and the points `p1` and `p2` and comparing the distances,
    /// because square root operations that are needed for computing the distances are avoided.
    #[inline]
    fn is_farther_from(self, p1: Point2<S>, p2: Point2<S>) -> bool {
        let (v1, v2) = (p1 - self, p2 - self);
        dot(v1, v1) > dot(v2, v2)
    }

    /// Checks which of the points `p1` and `p2` is closer to this point and returns the closest one.
    ///
    /// This is more computationally efficient than computing the distance between this point and the points `p1` and `p2` and comparing the distances,
    /// because square root operations that are needed for computing the distances are avoided.
    #[inline]
    fn closest(self, p1: Point2<S>, p2: Point2<S>) -> Point2<S> {
        if self.is_closer_to(p1, p2) { p1 } else { p2 }
    }

    /// Checks which of the points `p1` and `p2` is farther from this point and returns the closest one.
    ///
    /// This is more computationally efficient than computing the distance between this point and the points `p1` and `p2` and comparing the distances,
    /// because square root operations that are needed for computing the distances are avoided.
    #[inline]
    fn farthest(self, p1: Point2<S>, p2: Point2<S>) -> Point2<S> {
        if self.is_farther_from(p1, p2) { p1 } else { p2 }
    }
}

impl<S: Scalar> Add<Vector2<S>> for Point2<S> {
    type Output = Point2<S>;

    #[inline]
    fn add(self, v: Vector2<S>) -> Point2<S> {
        Point2::new(self.x + v.x, self.y + v.y)
    }
}

impl<S: Scalar> AddAssign<Vector2<S>> for Point2<S> {
    #[inline]
    fn add_assign(&mut self, v: Vector2<S>) {
        self.x += v.x;
        self.y += v.y;
    }
}

impl<S: Scalar> Sub<Vector2<S>> for Point2<S> {
    type Output = Point2<S>;

    #[inline]
    fn sub(self, v: Vector2<S>) -> Point2<S> {
        Point2::new(self.x - v.x, self.y - v.y)
    }
}

impl<S: Scalar> SubAssign<Vector2<S>> for Point2<S> {
    #[inline]
    fn sub_assign(&mut self, v: Vector2<S>) {
        self.x -= v.x;
        self.y -= v.y;
    }
}

impl<S: Scalar> Sub<Point2<S>> for Point2<S> {
    type Output = Vector2<S>;

    #[inline]
    fn sub(self, p: Point2<S>) -> Vector2<S> {
        Vector2::new(self.x - p.x, self.y - p.y)
    }
}

impl<S: Scalar> Neg for Point2<S> {
    type Output = Point2<S>;

    #[inline]
    fn neg(self) -> Point2<S> {
        Point2::new(-self.x, -self.y)
    }
}

impl<S: Scalar> Mul<S> for Point2<S> {
    type Output = Point2<S>;

    #[inline]
    fn mul(self, s: S) -> Point2<S> {
        Point2::new(self.x * s, self.y * s)
    }
}

impl<S: Scalar> MulAssign<S> for Point2<S> {
    #[inline]
    fn mul_assign(&mut self, s: S) {
        self.x *= s;
        self.y *= s;
    }
}

impl Mul<Point2f> for f32 {
    type Output = Point2f;

    #[inline]
    fn mul(self, p: Point2f) -> Point2f {
        p * self
    }
}

impl Mul<Point2d> for f64 {
    type Output = Point2d;

    #[inline]
    fn mul(self, p: Point2d) -> Point2d {
        p * self
    }
}

impl<S: Scalar> Div<S> for Point2<S> {
    type Output = Point2<S>;

    #[inline]
    fn div(self, s: S) -> Point2<S> {
        Point2::new(self.x / s, self.y / s)
    }
}

impl<S: Scalar> DivAssign<S> for Point2<S> {
    #[inline]
    fn div_assign(&mut self, s: S) {
        self.x /= s;
        self.y /= s;
    }
}

impl<S: Scalar> From<Vector2<S>> for Point2<S> {
    #[inline]
    fn from(v: Vector2<S>) -> Self {
        Point2::new(v.x, v.y)
    }
}
