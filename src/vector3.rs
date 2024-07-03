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

use crate::{CrossProduct, Dimension3, dot, DotProduct, Length, max, min, MinMax, Point3, RelativeLength, Scalar};

/// Vector in 3D space.
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Vector3<S: Scalar> {
    pub x: S,
    pub y: S,
    pub z: S,
}

/// Alias for `Vector3<f32>`.
pub type Vector3f = Vector3<f32>;

/// Alias for `Vector3<f64>`.
pub type Vector3d = Vector3<f64>;

#[inline]
pub fn vector3<S: Scalar>(x: S, y: S, z: S) -> Vector3<S> {
    Vector3::new(x, y, z)
}

#[inline]
pub fn vector3f(x: f32, y: f32, z: f32) -> Vector3f {
    Vector3f::new(x, y, z)
}

#[inline]
pub fn vector3d(x: f64, y: f64, z: f64) -> Vector3d {
    Vector3d::new(x, y, z)
}

// ===== Vector3 ===============================================================================================================================================

impl<S: Scalar> Vector3<S> {
    /// Creates and returns a new `Vector3` with x, y and z coordinates.
    #[inline]
    pub fn new(x: S, y: S, z: S) -> Vector3<S> {
        Vector3 { x, y, z }
    }

    /// Returns a `Vector3` which represents the zero vector (x = 0, y = 0 and z = 0).
    #[inline]
    pub fn zero() -> Vector3<S> {
        Vector3::new(S::zero(), S::zero(), S::zero())
    }

    /// Returns a `Vector3` of length 1 which represents the X axis (x = 1, y = 0 and z = 0).
    #[inline]
    pub fn x_axis() -> Vector3<S> {
        Vector3::new(S::one(), S::zero(), S::zero())
    }

    /// Returns a `Vector3` of length 1 which represents the Y axis (x = 0, y = 1 and z = 0).
    #[inline]
    pub fn y_axis() -> Vector3<S> {
        Vector3::new(S::zero(), S::one(), S::zero())
    }

    /// Returns a `Vector3` of length 1 which represents the Z axis (x = 0, y = 0 and z = 1).
    #[inline]
    pub fn z_axis() -> Vector3<S> {
        Vector3::new(S::zero(), S::zero(), S::one())
    }

    /// Returns a `Vector3` of length 1 which represents the axis specified by a dimension.
    #[inline]
    pub fn axis(dim: Dimension3) -> Vector3<S> {
        match dim {
            Dimension3::X => Vector3::x_axis(),
            Dimension3::Y => Vector3::y_axis(),
            Dimension3::Z => Vector3::z_axis(),
        }
    }

    /// Creates and returns a new `Vector3` which points in the same direction as this vector, but with length 1.
    #[inline]
    pub fn normalize(self) -> Vector3<S> {
        self / self.length()
    }

    /// Returns the dimension with the smallest extent of this vector.
    #[inline]
    pub fn min_dimension(self) -> Dimension3 {
        let Vector3 { x, y, z } = self.abs();
        if x <= y && x <= z { Dimension3::X } else if y <= z { Dimension3::Y } else { Dimension3::Z }
    }

    /// Returns the dimension with the largest extent of this vector.
    #[inline]
    pub fn max_dimension(self) -> Dimension3 {
        let Vector3 { x, y, z } = self.abs();
        if x > y && x > z { Dimension3::X } else if y > z { Dimension3::Y } else { Dimension3::Z }
    }

    /// Returns the element-wise floor of this vector.
    #[inline]
    pub fn floor(self) -> Vector3<S> {
        Vector3::new(self.x.floor(), self.y.floor(), self.z.floor())
    }

    /// Returns the element-wise ceiling of this vector.
    #[inline]
    pub fn ceil(self) -> Vector3<S> {
        Vector3::new(self.x.ceil(), self.y.ceil(), self.z.ceil())
    }

    /// Returns the element-wise rounded value of this vector.
    #[inline]
    pub fn round(self) -> Vector3<S> {
        Vector3::new(self.x.round(), self.y.round(), self.z.round())
    }

    /// Returns the element-wise truncated value of this vector.
    #[inline]
    pub fn trunc(self) -> Vector3<S> {
        Vector3::new(self.x.trunc(), self.y.trunc(), self.z.trunc())
    }

    /// Returns the element-wise fractional value of this vector.
    #[inline]
    pub fn fract(self) -> Vector3<S> {
        Vector3::new(self.x.fract(), self.y.fract(), self.z.fract())
    }

    /// Returns the element-wise absolute value of this vector.
    #[inline]
    pub fn abs(self) -> Vector3<S> {
        Vector3::new(self.x.abs(), self.y.abs(), self.z.abs())
    }

    /// Returns a point with a permutation of the elements of this vector.
    #[inline]
    pub fn permute(self, dim_x: Dimension3, dim_y: Dimension3, dim_z: Dimension3) -> Vector3<S> {
        Vector3::new(self[dim_x], self[dim_y], self[dim_z])
    }
}

impl<S: Scalar> Index<Dimension3> for Vector3<S> {
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

impl<S: Scalar> IndexMut<Dimension3> for Vector3<S> {
    #[inline]
    fn index_mut(&mut self, dim: Dimension3) -> &mut S {
        match dim {
            Dimension3::X => &mut self.x,
            Dimension3::Y => &mut self.y,
            Dimension3::Z => &mut self.z,
        }
    }
}

impl<S: Scalar> MinMax for Vector3<S> {
    /// Returns the element-wise minimum of two vectors.
    #[inline]
    fn min(self, v: Vector3<S>) -> Vector3<S> {
        Vector3::new(min(self.x, v.x), min(self.y, v.y), min(self.z, v.z))
    }

    /// Returns the element-wise maximum of two vectors.
    #[inline]
    fn max(self, v: Vector3<S>) -> Vector3<S> {
        Vector3::new(max(self.x, v.x), max(self.y, v.y), max(self.z, v.z))
    }
}

impl<S: Scalar> Length for Vector3<S> {
    type Output = S;

    /// Computes and returns the length of a vector.
    #[inline]
    fn length(self) -> S {
        S::sqrt(dot(self, self))
    }
}

impl<S: Scalar> RelativeLength for Vector3<S> {
    /// Returns `true` if this vector is shorter than the other vector.
    ///
    /// This is more computationally efficient than computing the lengths of the vectors and comparing them,
    /// because square root operations that are needed for computing the lengths are avoided.
    #[inline]
    fn is_shorter_than(self, v: Vector3<S>) -> bool {
        dot(self, self) < dot(v, v)
    }

    /// Returns `true` if this vector is longer than the other vector.
    ///
    /// This is more computationally efficient than computing the lengths of the vectors and comparing them,
    /// because square root operations that are needed for computing the lengths are avoided.
    #[inline]
    fn is_longer_than(self, v: Vector3<S>) -> bool {
        dot(self, self) > dot(v, v)
    }

    /// Returns the shortest of two vectors.
    ///
    /// This is more computationally efficient than computing the lengths of the vectors and comparing them,
    /// because square root operations that are needed for computing the lengths are avoided.
    #[inline]
    fn shortest(self, v: Vector3<S>) -> Vector3<S> {
        if self.is_shorter_than(v) { self } else { v }
    }

    /// Returns the longest of two vectors.
    ///
    /// This is more computationally efficient than computing the lengths of the vectors and comparing them,
    /// because square root operations that are needed for computing the lengths are avoided.
    #[inline]
    fn longest(self, v: Vector3<S>) -> Vector3<S> {
        if self.is_longer_than(v) { self } else { v }
    }
}

impl<S: Scalar> DotProduct<Vector3<S>> for Vector3<S> {
    type Output = S;

    /// Computes and returns the dot product between two vectors.
    #[inline]
    fn dot(self, v: Vector3<S>) -> S {
        self.x * v.x + self.y * v.y + self.z * v.z
    }
}

impl<S: Scalar> CrossProduct<Vector3<S>> for Vector3<S> {
    type Output = Vector3<S>;

    /// Computes and returns the cross product between two vectors.
    #[inline]
    fn cross(self, v: Vector3<S>) -> Vector3<S> {
        Vector3::new(self.y * v.z - self.z * v.y, self.z * v.x - self.x * v.z, self.x * v.y - self.y * v.x)
    }
}

impl<S: Scalar> Add<Vector3<S>> for Vector3<S> {
    type Output = Vector3<S>;

    #[inline]
    fn add(self, v: Vector3<S>) -> Vector3<S> {
        Vector3::new(self.x + v.x, self.y + v.y, self.z + v.z)
    }
}

impl<S: Scalar> AddAssign<Vector3<S>> for Vector3<S> {
    #[inline]
    fn add_assign(&mut self, v: Vector3<S>) {
        self.x += v.x;
        self.y += v.y;
        self.z += v.z;
    }
}

impl<S: Scalar> Sub<Vector3<S>> for Vector3<S> {
    type Output = Vector3<S>;

    #[inline]
    fn sub(self, v: Vector3<S>) -> Vector3<S> {
        Vector3::new(self.x - v.x, self.y - v.y, self.z - v.z)
    }
}

impl<S: Scalar> SubAssign<Vector3<S>> for Vector3<S> {
    #[inline]
    fn sub_assign(&mut self, v: Vector3<S>) {
        self.x -= v.x;
        self.y -= v.y;
        self.z -= v.z;
    }
}

impl<S: Scalar> Neg for Vector3<S> {
    type Output = Vector3<S>;

    #[inline]
    fn neg(self) -> Vector3<S> {
        Vector3::new(-self.x, -self.y, -self.z)
    }
}

impl<S: Scalar> Mul<S> for Vector3<S> {
    type Output = Vector3<S>;

    #[inline]
    fn mul(self, s: S) -> Vector3<S> {
        Vector3::new(self.x * s, self.y * s, self.z * s)
    }
}

impl<S: Scalar> MulAssign<S> for Vector3<S> {
    #[inline]
    fn mul_assign(&mut self, s: S) {
        self.x *= s;
        self.y *= s;
        self.z *= s;
    }
}

impl Mul<Vector3f> for f32 {
    type Output = Vector3f;

    #[inline]
    fn mul(self, v: Vector3f) -> Vector3f {
        v * self
    }
}

impl Mul<Vector3d> for f64 {
    type Output = Vector3d;

    #[inline]
    fn mul(self, v: Vector3d) -> Vector3d {
        v * self
    }
}

impl<S: Scalar> Div<S> for Vector3<S> {
    type Output = Vector3<S>;

    #[inline]
    fn div(self, s: S) -> Vector3<S> {
        Vector3::new(self.x / s, self.y / s, self.z / s)
    }
}

impl<S: Scalar> DivAssign<S> for Vector3<S> {
    #[inline]
    fn div_assign(&mut self, s: S) {
        self.x /= s;
        self.y /= s;
        self.z /= s;
    }
}

impl<S: Scalar> From<Point3<S>> for Vector3<S> {
    #[inline]
    fn from(p: Point3<S>) -> Vector3<S> {
        Vector3::new(p.x, p.y, p.z)
    }
}
