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

use crate::{Dimension2, dot, DotProduct, Length, max, min, MinMax, Point2, RelativeLength, Scalar};

/// Vector in 2D space.
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Vector2<S: Scalar> {
    pub x: S,
    pub y: S,
}

/// Alias for `Vector2<f32>`.
pub type Vector2f = Vector2<f32>;

/// Alias for `Vector2<f64>`.
pub type Vector2d = Vector2<f64>;

#[inline]
pub fn vector2<S: Scalar>(x: S, y: S) -> Vector2<S> {
    Vector2::new(x, y)
}

#[inline]
pub fn vector2f(x: f32, y: f32) -> Vector2f {
    Vector2f::new(x, y)
}

#[inline]
pub fn vector2d(x: f64, y: f64) -> Vector2d {
    Vector2d::new(x, y)
}

// ===== Vector2 ===============================================================================================================================================

impl<S: Scalar> Vector2<S> {
    /// Creates and returns a new `Vector2` with x and y coordinates.
    #[inline]
    pub fn new(x: S, y: S) -> Vector2<S> {
        Vector2 { x, y }
    }

    /// Returns a `Vector2` which represents the zero vector (x = 0 and y = 0).
    #[inline]
    pub fn zero() -> Vector2<S> {
        Vector2::new(S::zero(), S::zero())
    }

    /// Returns a `Vector2` of length 1 which represents the X axis (x = 1 and y = 0).
    #[inline]
    pub fn x_axis() -> Vector2<S> {
        Vector2::new(S::one(), S::zero())
    }

    /// Returns a `Vector2` of length 1 which represents the Y axis (x = 0 and y = 1).
    #[inline]
    pub fn y_axis() -> Vector2<S> {
        Vector2::new(S::zero(), S::one())
    }

    /// Returns a `Vector2` of length 1 which represents the axis specified by a dimension.
    #[inline]
    pub fn axis(dim: Dimension2) -> Vector2<S> {
        match dim {
            Dimension2::X => Vector2::x_axis(),
            Dimension2::Y => Vector2::y_axis(),
        }
    }

    /// Creates and returns a new `Vector2` which points in the same direction as this vector, but with length 1.
    #[inline]
    pub fn normalize(self) -> Vector2<S> {
        self / self.length()
    }

    /// Returns the dimension with the smallest extent of this vector.
    #[inline]
    pub fn min_dimension(self) -> Dimension2 {
        let Vector2 { x, y } = self.abs();
        if x <= y { Dimension2::X } else { Dimension2::Y }
    }

    /// Returns the dimension with the largest extent of this vector.
    #[inline]
    pub fn max_dimension(self) -> Dimension2 {
        let Vector2 { x, y } = self.abs();
        if x > y { Dimension2::X } else { Dimension2::Y }
    }

    /// Returns the element-wise floor of this vector.
    #[inline]
    pub fn floor(self) -> Vector2<S> {
        Vector2::new(self.x.floor(), self.y.floor())
    }

    /// Returns the element-wise ceiling of this vector.
    #[inline]
    pub fn ceil(self) -> Vector2<S> {
        Vector2::new(self.x.ceil(), self.y.ceil())
    }

    /// Returns the element-wise rounded value of this vector.
    #[inline]
    pub fn round(self) -> Vector2<S> {
        Vector2::new(self.x.round(), self.y.round())
    }

    /// Returns the element-wise truncated value of this vector.
    #[inline]
    pub fn trunc(self) -> Vector2<S> {
        Vector2::new(self.x.trunc(), self.y.trunc())
    }

    /// Returns the element-wise fractional value of this vector.
    #[inline]
    pub fn fract(self) -> Vector2<S> {
        Vector2::new(self.x.fract(), self.y.fract())
    }

    /// Returns the element-wise absolute value of this vector.
    #[inline]
    pub fn abs(self) -> Vector2<S> {
        Vector2::new(self.x.abs(), self.y.abs())
    }

    /// Returns a point with a permutation of the elements of this vector.
    #[inline]
    pub fn permute(self, dim_x: Dimension2, dim_y: Dimension2) -> Vector2<S> {
        Vector2::new(self[dim_x], self[dim_y])
    }
}

impl<S: Scalar> Index<Dimension2> for Vector2<S> {
    type Output = S;

    #[inline]
    fn index(&self, dim: Dimension2) -> &S {
        match dim {
            Dimension2::X => &self.x,
            Dimension2::Y => &self.y,
        }
    }
}

impl<S: Scalar> IndexMut<Dimension2> for Vector2<S> {
    #[inline]
    fn index_mut(&mut self, dim: Dimension2) -> &mut S {
        match dim {
            Dimension2::X => &mut self.x,
            Dimension2::Y => &mut self.y,
        }
    }
}

impl<S: Scalar> MinMax for Vector2<S> {
    /// Returns the element-wise minimum of two vectors.
    #[inline]
    fn min(self, v: Vector2<S>) -> Vector2<S> {
        Vector2::new(min(self.x, v.x), min(self.y, v.y))
    }

    /// Returns the element-wise maximum of two vectors.
    #[inline]
    fn max(self, v: Vector2<S>) -> Vector2<S> {
        Vector2::new(max(self.x, v.x), max(self.y, v.y))
    }
}

impl<S: Scalar> Length for Vector2<S> {
    type Output = S;

    /// Computes and returns the length of this vector.
    #[inline]
    fn length(self) -> S {
        S::sqrt(dot(self, self))
    }
}

impl<S: Scalar> RelativeLength for Vector2<S> {
    /// Returns `true` if this vector is shorter than the other vector.
    ///
    /// This is more computationally efficient than computing the lengths of the vectors and comparing them,
    /// because square root operations that are needed for computing the lengths are avoided.
    #[inline]
    fn is_shorter_than(self, v: Vector2<S>) -> bool {
        dot(self, self) < dot(v, v)
    }

    /// Returns `true` if this vector is longer than the other vector.
    ///
    /// This is more computationally efficient than computing the lengths of the vectors and comparing them,
    /// because square root operations that are needed for computing the lengths are avoided.
    #[inline]
    fn is_longer_than(self, v: Vector2<S>) -> bool {
        dot(self, self) > dot(v, v)
    }

    /// Returns the shortest of two vectors.
    ///
    /// This is more computationally efficient than computing the lengths of the vectors and comparing them,
    /// because square root operations that are needed for computing the lengths are avoided.
    #[inline]
    fn shortest(self, v: Vector2<S>) -> Vector2<S> {
        if self.is_shorter_than(v) { self } else { v }
    }

    /// Returns the longest of two vectors.
    ///
    /// This is more computationally efficient than computing the lengths of the vectors and comparing them,
    /// because square root operations that are needed for computing the lengths are avoided.
    #[inline]
    fn longest(self, v: Vector2<S>) -> Vector2<S> {
        if self.is_longer_than(v) { self } else { v }
    }
}

impl<S: Scalar> DotProduct<Vector2<S>> for Vector2<S> {
    type Output = S;

    /// Computes and returns the dot product between two vectors.
    #[inline]
    fn dot(self, v: Vector2<S>) -> S {
        self.x * v.x + self.y * v.y
    }
}

impl<S: Scalar> Add<Vector2<S>> for Vector2<S> {
    type Output = Vector2<S>;

    #[inline]
    fn add(self, v: Vector2<S>) -> Vector2<S> {
        Vector2::new(self.x + v.x, self.y + v.y)
    }
}

impl<S: Scalar> AddAssign<Vector2<S>> for Vector2<S> {
    #[inline]
    fn add_assign(&mut self, v: Vector2<S>) {
        self.x += v.x;
        self.y += v.y;
    }
}

impl<S: Scalar> Sub<Vector2<S>> for Vector2<S> {
    type Output = Vector2<S>;

    #[inline]
    fn sub(self, v: Vector2<S>) -> Vector2<S> {
        Vector2::new(self.x - v.x, self.y - v.y)
    }
}

impl<S: Scalar> SubAssign<Vector2<S>> for Vector2<S> {
    #[inline]
    fn sub_assign(&mut self, v: Vector2<S>) {
        self.x -= v.x;
        self.y -= v.y;
    }
}

impl<S: Scalar> Neg for Vector2<S> {
    type Output = Vector2<S>;

    #[inline]
    fn neg(self) -> Vector2<S> {
        Vector2::new(-self.x, -self.y)
    }
}

impl<S: Scalar> Mul<S> for Vector2<S> {
    type Output = Vector2<S>;

    #[inline]
    fn mul(self, s: S) -> Vector2<S> {
        Vector2::new(self.x * s, self.y * s)
    }
}

impl<S: Scalar> MulAssign<S> for Vector2<S> {
    #[inline]
    fn mul_assign(&mut self, s: S) {
        self.x *= s;
        self.y *= s;
    }
}

impl Mul<Vector2f> for f32 {
    type Output = Vector2f;

    #[inline]
    fn mul(self, v: Vector2f) -> Vector2f {
        v * self
    }
}

impl Mul<Vector2d> for f64 {
    type Output = Vector2d;

    #[inline]
    fn mul(self, v: Vector2d) -> Vector2d {
        v * self
    }
}

impl<S: Scalar> Div<S> for Vector2<S> {
    type Output = Vector2<S>;

    #[inline]
    fn div(self, s: S) -> Vector2<S> {
        Vector2::new(self.x / s, self.y / s)
    }
}

impl<S: Scalar> DivAssign<S> for Vector2<S> {
    #[inline]
    fn div_assign(&mut self, s: S) {
        self.x /= s;
        self.y /= s;
    }
}

impl<S: Scalar> From<Point2<S>> for Vector2<S> {
    #[inline]
    fn from(p: Point2<S>) -> Vector2<S> {
        Vector2::new(p.x, p.y)
    }
}
