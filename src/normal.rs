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

use crate::{Dimension3, DotProduct, Length, MinMax, Point3, RelativeLength, Scalar, Vector3};
use num_traits::{ConstZero, Zero};
use std::cmp::Ordering;
use std::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign};

#[cfg(feature = "rand")]
use rand::{distributions::Standard, prelude::Distribution, Rng};

/// Surface normal in 3D space.
///
/// A normal is similar to a vector, but there are a number of differences. One main difference is that transforming a normal is done differently
/// than transforming a vector; see the implementation of [Transform](crate::Transform) for [Transform3](crate::Transform3) with [Normal3].
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Normal3<S: Scalar> {
    pub x: S,
    pub y: S,
    pub z: S,
}

// ===== Normal3 ===============================================================================================================================================

impl<S: Scalar> Normal3<S> {
    /// Normal that represents the X axis: (1, 0, 0).
    pub const X_AXIS: Normal3<S> = Normal3 { x: S::ONE, y: S::ZERO, z: S::ZERO };

    /// Normal that represents the Y axis: (0, 1, 0).
    pub const Y_AXIS: Normal3<S> = Normal3 { x: S::ZERO, y: S::ONE, z: S::ZERO };

    /// Normal that represents the Z axis: (0, 0, 1).
    pub const Z_AXIS: Normal3<S> = Normal3 { x: S::ZERO, y: S::ZERO, z: S::ONE };

    /// Creates and returns a new normal.
    pub fn new(x: S, y: S, z: S) -> Normal3<S> {
        Normal3 { x, y, z }
    }

    /// Returns a normal that represents the X axis.
    pub fn x_axis() -> Normal3<S> {
        Normal3::X_AXIS
    }

    /// Returns a normal that represents the Y axis.
    pub fn y_axis() -> Normal3<S> {
        Normal3::Y_AXIS
    }

    /// Returns a normal that represents the Z axis.
    pub fn z_axis() -> Normal3<S> {
        Normal3::Z_AXIS
    }

    /// Returns a normal that represents the axis corresponding to `dimension`.
    ///
    /// # Example
    /// ```
    /// use vecmath::{Dimension3, Normal3};
    ///
    /// let x_axis: Normal3<f32> = Normal3::axis(Dimension3::X);
    /// println!("{:?}", x_axis); // prints: Normal3 { x: 1.0, y: 0.0, z: 0.0 }
    /// ```
    pub fn axis(dimension: Dimension3) -> Normal3<S> {
        match dimension {
            Dimension3::X => Normal3::X_AXIS,
            Dimension3::Y => Normal3::Y_AXIS,
            Dimension3::Z => Normal3::Z_AXIS,
        }
    }

    /// Returns a normal that points in the same direction as this normal but with length 1.
    pub fn normalized(self) -> Normal3<S> {
        self / self.length()
    }

    /// Changes this normal so that the length is 1.
    pub fn normalize(&mut self) {
        *self /= self.length();
    }

    /// Returns the dimension with the smallest extent of this normal.
    pub fn min_dimension(self) -> Dimension3 {
        let Normal3 { x, y, z } = self.abs();
        if x <= y && x <= z { Dimension3::X } else if y <= z { Dimension3::Y } else { Dimension3::Z }
    }

    /// Returns the dimension with the largest extent of this normal.
    pub fn max_dimension(self) -> Dimension3 {
        let Normal3 { x, y, z } = self.abs();
        if x > y && x > z { Dimension3::X } else if y > z { Dimension3::Y } else { Dimension3::Z }
    }

    /// Returns the element-wise floor of this normal.
    pub fn floor(self) -> Normal3<S> {
        Normal3 {
            x: self.x.floor(),
            y: self.y.floor(),
            z: self.z.floor(),
        }
    }

    /// Returns the element-wise ceiling of this normal.
    pub fn ceil(self) -> Normal3<S> {
        Normal3 {
            x: self.x.ceil(),
            y: self.y.ceil(),
            z: self.z.ceil(),
        }
    }

    /// Returns the element-wise rounded value of this normal.
    pub fn round(self) -> Normal3<S> {
        Normal3 {
            x: self.x.round(),
            y: self.y.round(),
            z: self.z.round(),
        }
    }

    /// Returns the element-wise truncated value of this normal.
    pub fn trunc(self) -> Normal3<S> {
        Normal3 {
            x: self.x.trunc(),
            y: self.y.trunc(),
            z: self.z.trunc(),
        }
    }

    /// Returns the element-wise fractional value of this normal.
    pub fn fract(self) -> Normal3<S> {
        Normal3 {
            x: self.x.fract(),
            y: self.y.fract(),
            z: self.z.fract(),
        }
    }

    /// Returns the element-wise absolute value of this normal.
    pub fn abs(self) -> Normal3<S> {
        Normal3 {
            x: self.x.abs(),
            y: self.y.abs(),
            z: self.z.abs(),
        }
    }

    /// Returns a normal with a permutation of the elements of this normal.
    pub fn permutation(self, dim_x: Dimension3, dim_y: Dimension3, dim_z: Dimension3) -> Normal3<S> {
        Normal3 {
            x: self[dim_x],
            y: self[dim_y],
            z: self[dim_z],
        }
    }
}

impl<S: Scalar> Index<Dimension3> for Normal3<S> {
    type Output = S;

    fn index(&self, index: Dimension3) -> &S {
        match index {
            Dimension3::X => &self.x,
            Dimension3::Y => &self.y,
            Dimension3::Z => &self.z,
        }
    }
}

impl<S: Scalar> IndexMut<Dimension3> for Normal3<S> {
    fn index_mut(&mut self, index: Dimension3) -> &mut S {
        match index {
            Dimension3::X => &mut self.x,
            Dimension3::Y => &mut self.y,
            Dimension3::Z => &mut self.z,
        }
    }
}

impl<S: Scalar> Zero for Normal3<S> {
    /// Returns the zero normal: (0, 0, 0).
    fn zero() -> Normal3<S> {
        Normal3::ZERO
    }

    /// Returns `true` if this normal is equal to the zero normal, `false` otherwise.
    fn is_zero(&self) -> bool {
        self.x.is_zero() && self.y.is_zero() && self.z.is_zero()
    }
}

impl<S: Scalar> ConstZero for Normal3<S> {
    /// The zero normal: (0, 0, 0).
    const ZERO: Normal3<S> = Normal3 { x: S::ZERO, y: S::ZERO, z: S::ZERO };
}

impl<S: Scalar> Add for Normal3<S> {
    type Output = Normal3<S>;

    fn add(self, other: Normal3<S>) -> Normal3<S> {
        Normal3 {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

impl<S: Scalar> AddAssign for Normal3<S> {
    fn add_assign(&mut self, other: Normal3<S>) {
        self.x += other.x;
        self.y += other.y;
        self.z += other.z;
    }
}

impl<S: Scalar> Sub for Normal3<S> {
    type Output = Normal3<S>;

    fn sub(self, other: Normal3<S>) -> Normal3<S> {
        Normal3 {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

impl<S: Scalar> SubAssign for Normal3<S> {
    fn sub_assign(&mut self, other: Normal3<S>) {
        self.x -= other.x;
        self.y -= other.y;
        self.z -= other.z;
    }
}

impl<S: Scalar> Neg for Normal3<S> {
    type Output = Normal3<S>;

    fn neg(self) -> Normal3<S> {
        Normal3 {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

impl<S: Scalar> Mul<S> for Normal3<S> {
    type Output = Normal3<S>;

    fn mul(self, value: S) -> Normal3<S> {
        Normal3 {
            x: self.x * value,
            y: self.y * value,
            z: self.z * value,
        }
    }
}

impl<S: Scalar> MulAssign<S> for Normal3<S> {
    fn mul_assign(&mut self, value: S) {
        self.x *= value;
        self.y *= value;
        self.z *= value;
    }
}

impl<S: Scalar> Div<S> for Normal3<S> {
    type Output = Normal3<S>;

    fn div(self, value: S) -> Normal3<S> {
        Normal3 {
            x: self.x / value,
            y: self.y / value,
            z: self.z / value,
        }
    }
}

impl<S: Scalar> DivAssign<S> for Normal3<S> {
    fn div_assign(&mut self, value: S) {
        self.x /= value;
        self.y /= value;
        self.z /= value;
    }
}

impl<S: Scalar> Rem<S> for Normal3<S> {
    type Output = Normal3<S>;

    fn rem(self, value: S) -> Normal3<S> {
        Normal3 {
            x: self.x % value,
            y: self.y % value,
            z: self.z % value,
        }
    }
}

impl<S: Scalar> RemAssign<S> for Normal3<S> {
    fn rem_assign(&mut self, value: S) {
        self.x %= value;
        self.y %= value;
        self.z %= value;
    }
}

impl<S: Scalar> MinMax for Normal3<S> {
    /// Returns the element-wise minimum of two normals.
    fn min(self, other: Self) -> Self {
        Normal3 {
            x: self.x.min(other.x),
            y: self.y.min(other.y),
            z: self.z.min(other.z),
        }
    }

    /// Returns the element-wise maximum of two normals.
    fn max(self, other: Self) -> Self {
        Normal3 {
            x: self.x.max(other.x),
            y: self.y.max(other.y),
            z: self.z.max(other.z),
        }
    }

    /// Returns the element-wise minimum and maximum of two normals.
    fn min_max(self, other: Self) -> (Self, Self) {
        let (min_x, max_x) = self.x.min_max(other.x);
        let (min_y, max_y) = self.y.min_max(other.y);
        let (min_z, max_z) = self.z.min_max(other.z);

        (Normal3 { x: min_x, y: min_y, z: min_z }, Normal3 { x: max_x, y: max_y, z: max_z })
    }
}

impl<S: Scalar> Length<S> for Normal3<S> {
    /// Computes and returns the length of this normal.
    ///
    /// Note: If you only need to compare the lengths of normals, use the methods of trait [RelativeLength] instead of calling this method.
    /// Computing the length of a normal involves a relatively expensive square root operation that can be avoided if you only need to
    /// compare the length of a normal to the length of another normal.
    fn length(self) -> S {
        self.dot(self).sqrt()
    }
}

impl<S: Scalar> RelativeLength<S> for Normal3<S> {
    fn cmp_length(self, other: Normal3<S>) -> Ordering {
        self.dot(self).total_cmp(&other.dot(other))
    }
}

impl<S: Scalar> DotProduct<S> for Normal3<S> {
    /// Computes and returns the dot product of this normal and another normal.
    fn dot(self, other: Normal3<S>) -> S {
        self.x * other.x + self.y * other.y + self.z * other.z
    }
}

impl<S: Scalar> DotProduct<S, Vector3<S>> for Normal3<S> {
    /// Computes and returns the dot product of this normal and a vector.
    fn dot(self, vector: Vector3<S>) -> S {
        self.x * vector.x + self.y * vector.y + self.z * vector.z
    }
}

impl<S: Scalar> From<Vector3<S>> for Normal3<S> {
    fn from(vector: Vector3<S>) -> Self {
        Normal3 {
            x: vector.x,
            y: vector.y,
            z: vector.z,
        }
    }
}

impl<S: Scalar> From<Point3<S>> for Normal3<S> {
    fn from(point: Point3<S>) -> Self {
        Normal3 {
            x: point.x,
            y: point.y,
            z: point.z,
        }
    }
}

#[cfg(feature = "rand")]
impl<S: Scalar> Distribution<Normal3<S>> for Standard
where
    Standard: Distribution<(S, S, S)>,
{
    /// Generates a random vector with elements in the range `0..1` (uniformly distributed).
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Normal3<S> {
        let (x, y, z) = rng.gen();
        Normal3 { x, y, z }
    }
}
