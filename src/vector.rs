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

use crate::{Dimension2, Dimension3, DotProduct, Length, MinMax, Normal3, Point2, Point3, RelativeLength, Scalar};
use num_traits::{ConstZero, Zero};
use std::cmp::Ordering;
use std::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign};

#[cfg(feature = "rand")]
use rand::{distributions::Standard, prelude::Distribution, Rng};

/// Vector in 2D space.
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Vector2<S: Scalar> {
    pub x: S,
    pub y: S,
}

/// Vector in 3D space.
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Vector3<S: Scalar> {
    pub x: S,
    pub y: S,
    pub z: S,
}

// ===== Vector2 ===============================================================================================================================================

impl<S: Scalar> Vector2<S> {
    /// Vector that represents the X axis: (1, 0).
    pub const X_AXIS: Vector2<S> = Vector2 { x: S::ONE, y: S::ZERO };

    /// Vector that represents the Y axis: (0, 1).
    pub const Y_AXIS: Vector2<S> = Vector2 { x: S::ZERO, y: S::ONE };

    /// Creates and returns a new vector.
    pub fn new(x: S, y: S) -> Vector2<S> {
        Vector2 { x, y }
    }

    /// Returns a vector that represents the X axis.
    pub fn x_axis() -> Vector2<S> {
        Vector2::X_AXIS
    }

    /// Returns a vector that represents the Y axis.
    pub fn y_axis() -> Vector2<S> {
        Vector2::Y_AXIS
    }

    /// Returns a vector that represents the axis corresponding to `dimension`.
    ///
    /// # Example
    /// ```
    /// use vecmath::{Dimension2, Vector2};
    ///
    /// let x_axis: Vector2<f32> = Vector2::axis(Dimension2::X);
    /// println!("{:?}", x_axis); // prints: Vector2 { x: 1.0, y: 0.0 }
    /// ```
    pub fn axis(dimension: Dimension2) -> Vector2<S> {
        match dimension {
            Dimension2::X => Vector2::X_AXIS,
            Dimension2::Y => Vector2::Y_AXIS,
        }
    }

    /// Returns a vector that points in the same direction as this vector but with length 1.
    pub fn normalized(self) -> Vector2<S> {
        self / self.length()
    }

    /// Changes this vector so that the length is 1.
    pub fn normalize(&mut self) {
        *self /= self.length();
    }

    /// Returns the dimension with the smallest extent of this vector.
    ///
    /// # Example
    /// ```
    /// use vecmath::Vector2;
    ///
    /// let v = Vector2::new(-2.0, 1.0);
    /// println!("{:?}", v.min_dimension()); // prints: Y
    /// ```
    pub fn min_dimension(self) -> Dimension2 {
        let Vector2 { x, y } = self.abs();
        if x <= y { Dimension2::X } else { Dimension2::Y }
    }

    /// Returns the dimension with the largest extent of this vector.
    ///
    /// # Example
    /// ```
    /// use vecmath::Vector2;
    ///
    /// let v = Vector2::new(-2.0, 1.0);
    /// println!("{:?}", v.max_dimension()); // prints: X
    /// ```
    pub fn max_dimension(self) -> Dimension2 {
        let Vector2 { x, y } = self.abs();
        if x > y { Dimension2::X } else { Dimension2::Y }
    }

    /// Returns the element-wise floor of this vector.
    pub fn floor(self) -> Vector2<S> {
        Vector2 {
            x: self.x.floor(),
            y: self.y.floor(),
        }
    }

    /// Returns the element-wise ceiling of this vector.
    pub fn ceil(self) -> Vector2<S> {
        Vector2 {
            x: self.x.ceil(),
            y: self.y.ceil(),
        }
    }

    /// Returns the element-wise rounded value of this vector.
    pub fn round(self) -> Vector2<S> {
        Vector2 {
            x: self.x.round(),
            y: self.y.round(),
        }
    }

    /// Returns the element-wise truncated value of this vector.
    pub fn trunc(self) -> Vector2<S> {
        Vector2 {
            x: self.x.trunc(),
            y: self.y.trunc(),
        }
    }

    /// Returns the element-wise fractional value of this vector.
    pub fn fract(self) -> Vector2<S> {
        Vector2 {
            x: self.x.fract(),
            y: self.y.fract(),
        }
    }

    /// Returns the element-wise absolute value of this vector.
    pub fn abs(self) -> Vector2<S> {
        Vector2 {
            x: self.x.abs(),
            y: self.y.abs(),
        }
    }

    /// Returns a vector with a permutation of the elements of this vector.
    pub fn permutation(self, dim_x: Dimension2, dim_y: Dimension2) -> Vector2<S> {
        Vector2 {
            x: self[dim_x],
            y: self[dim_y],
        }
    }
}

impl<S: Scalar> Index<Dimension2> for Vector2<S> {
    type Output = S;

    fn index(&self, index: Dimension2) -> &S {
        match index {
            Dimension2::X => &self.x,
            Dimension2::Y => &self.y,
        }
    }
}

impl<S: Scalar> IndexMut<Dimension2> for Vector2<S> {
    fn index_mut(&mut self, index: Dimension2) -> &mut S {
        match index {
            Dimension2::X => &mut self.x,
            Dimension2::Y => &mut self.y,
        }
    }
}

impl<S: Scalar> Zero for Vector2<S> {
    /// Returns the zero vector: (0, 0).
    fn zero() -> Vector2<S> {
        Vector2::ZERO
    }

    /// Returns `true` if this vector is equal to the zero vector, `false` otherwise.
    fn is_zero(&self) -> bool {
        self.x.is_zero() && self.y.is_zero()
    }
}

impl<S: Scalar> ConstZero for Vector2<S> {
    /// The zero vector: (0, 0).
    const ZERO: Vector2<S> = Vector2 { x: S::ZERO, y: S::ZERO };
}

impl<S: Scalar> Add for Vector2<S> {
    type Output = Vector2<S>;

    fn add(self, other: Vector2<S>) -> Vector2<S> {
        Vector2 {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

impl<S: Scalar> AddAssign for Vector2<S> {
    fn add_assign(&mut self, other: Vector2<S>) {
        self.x += other.x;
        self.y += other.y;
    }
}

impl<S: Scalar> Sub for Vector2<S> {
    type Output = Vector2<S>;

    fn sub(self, other: Vector2<S>) -> Vector2<S> {
        Vector2 {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }
}

impl<S: Scalar> SubAssign for Vector2<S> {
    fn sub_assign(&mut self, other: Vector2<S>) {
        self.x -= other.x;
        self.y -= other.y;
    }
}

impl<S: Scalar> Neg for Vector2<S> {
    type Output = Vector2<S>;

    fn neg(self) -> Vector2<S> {
        Vector2 {
            x: -self.x,
            y: -self.y,
        }
    }
}

impl<S: Scalar> Mul<S> for Vector2<S> {
    type Output = Vector2<S>;

    fn mul(self, value: S) -> Vector2<S> {
        Vector2 {
            x: self.x * value,
            y: self.y * value,
        }
    }
}

impl<S: Scalar> MulAssign<S> for Vector2<S> {
    fn mul_assign(&mut self, value: S) {
        self.x *= value;
        self.y *= value;
    }
}

impl<S: Scalar> Div<S> for Vector2<S> {
    type Output = Vector2<S>;

    fn div(self, value: S) -> Vector2<S> {
        Vector2 {
            x: self.x / value,
            y: self.y / value,
        }
    }
}

impl<S: Scalar> DivAssign<S> for Vector2<S> {
    fn div_assign(&mut self, value: S) {
        self.x /= value;
        self.y /= value;
    }
}

impl<S: Scalar> Rem<S> for Vector2<S> {
    type Output = Vector2<S>;

    fn rem(self, value: S) -> Vector2<S> {
        Vector2 {
            x: self.x % value,
            y: self.y % value,
        }
    }
}

impl<S: Scalar> RemAssign<S> for Vector2<S> {
    fn rem_assign(&mut self, value: S) {
        self.x %= value;
        self.y %= value;
    }
}

impl<S: Scalar> MinMax for Vector2<S> {
    /// Returns the element-wise minimum of two vectors.
    fn min(self, other: Self) -> Self {
        Vector2 {
            x: self.x.min(other.x),
            y: self.y.min(other.y),
        }
    }

    /// Returns the element-wise maximum of two vectors.
    fn max(self, other: Self) -> Self {
        Vector2 {
            x: self.x.max(other.x),
            y: self.y.max(other.y),
        }
    }

    /// Returns the element-wise minimum and maximum of two vectors.
    fn min_max(self, other: Self) -> (Self, Self) {
        let (min_x, max_x) = self.x.min_max(other.x);
        let (min_y, max_y) = self.y.min_max(other.y);

        (Vector2 { x: min_x, y: min_y }, Vector2 { x: max_x, y: max_y })
    }
}

impl<S: Scalar> Length<S> for Vector2<S> {
    /// Computes and returns the length of this vector.
    ///
    /// Note: If you only need to compare the lengths of vectors, use the methods of trait [RelativeLength] instead of calling this method.
    /// Computing the length of a vector involves a relatively expensive square root operation that can be avoided if you only need to
    /// compare the length of a vector to the length of another vector.
    fn length(self) -> S {
        self.dot(self).sqrt()
    }
}

impl<S: Scalar> RelativeLength<S> for Vector2<S> {
    fn cmp_length(self, other: Vector2<S>) -> Ordering {
        self.dot(self).total_cmp(&other.dot(other))
    }
}

impl<S: Scalar> DotProduct<S> for Vector2<S> {
    /// Computes and returns the dot product of this vector and another vector.
    fn dot(self, other: Vector2<S>) -> S {
        self.x * other.x + self.y * other.y
    }
}

impl<S: Scalar> From<Point2<S>> for Vector2<S> {
    fn from(point: Point2<S>) -> Self {
        Vector2 {
            x: point.x,
            y: point.y,
        }
    }
}

#[cfg(feature = "rand")]
impl<S: Scalar> Distribution<Vector2<S>> for Standard
where
    Standard: Distribution<(S, S)>,
{
    /// Generates a random vector with elements in the range `0..1` (uniformly distributed).
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Vector2<S> {
        let (x, y) = rng.gen();
        Vector2 { x, y }
    }
}

// ===== Vector3 ===============================================================================================================================================

impl<S: Scalar> Vector3<S> {
    /// Vector that represents the X axis: (1, 0, 0).
    pub const X_AXIS: Vector3<S> = Vector3 { x: S::ONE, y: S::ZERO, z: S::ZERO };

    /// Vector that represents the Y axis: (0, 1, 0).
    pub const Y_AXIS: Vector3<S> = Vector3 { x: S::ZERO, y: S::ONE, z: S::ZERO };

    /// Vector that represents the Z axis: (0, 0, 1).
    pub const Z_AXIS: Vector3<S> = Vector3 { x: S::ZERO, y: S::ZERO, z: S::ONE };

    /// Creates and returns a new vector.
    pub fn new(x: S, y: S, z: S) -> Vector3<S> {
        Vector3 { x, y, z }
    }

    /// Returns a vector that represents the X axis.
    pub fn x_axis() -> Vector3<S> {
        Vector3::X_AXIS
    }

    /// Returns a vector that represents the Y axis.
    pub fn y_axis() -> Vector3<S> {
        Vector3::Y_AXIS
    }

    /// Returns a vector that represents the Z axis.
    pub fn z_axis() -> Vector3<S> {
        Vector3::Z_AXIS
    }

    /// Returns a vector that represents the axis corresponding to `dimension`.
    ///
    /// # Example
    /// ```
    /// use vecmath::{Dimension3, Vector2, Vector3};
    ///
    /// let x_axis: Vector3<f32> = Vector3::axis(Dimension3::X);
    /// println!("{:?}", x_axis); // prints: Vector3 { x: 1.0, y: 0.0, z: 0.0 }
    /// ```
    pub fn axis(dimension: Dimension3) -> Vector3<S> {
        match dimension {
            Dimension3::X => Vector3::X_AXIS,
            Dimension3::Y => Vector3::Y_AXIS,
            Dimension3::Z => Vector3::Z_AXIS,
        }
    }

    /// Computes and returns the cross product of this vector and another vector.
    pub fn cross(self, other: Vector3<S>) -> Vector3<S> {
        Vector3 {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    /// Returns a vector that points in the same direction as this vector but with length 1.
    pub fn normalized(self) -> Vector3<S> {
        self / self.length()
    }

    /// Changes this vector so that the length is 1.
    pub fn normalize(&mut self) {
        *self /= self.length();
    }

    /// Returns the dimension with the smallest extent of this vector.
    ///
    /// # Example
    /// ```
    /// use vecmath::Vector3;
    ///
    /// let v = Vector3::new(-2.0, 1.0, 1.5);
    /// println!("{:?}", v.min_dimension()); // prints: Y
    /// ```
    pub fn min_dimension(self) -> Dimension3 {
        let Vector3 { x, y, z } = self.abs();
        if x <= y && x <= z { Dimension3::X } else if y <= z { Dimension3::Y } else { Dimension3::Z }
    }

    /// Returns the dimension with the largest extent of this vector.
    ///
    /// # Example
    /// ```
    /// use vecmath::Vector3;
    ///
    /// let v = Vector3::new(-2.0, 1.0, 1.5);
    /// println!("{:?}", v.max_dimension()); // prints: X
    /// ```
    pub fn max_dimension(self) -> Dimension3 {
        let Vector3 { x, y, z } = self.abs();
        if x > y && x > z { Dimension3::X } else if y > z { Dimension3::Y } else { Dimension3::Z }
    }

    /// Returns the element-wise floor of this vector.
    pub fn floor(self) -> Vector3<S> {
        Vector3 {
            x: self.x.floor(),
            y: self.y.floor(),
            z: self.z.floor(),
        }
    }

    /// Returns the element-wise ceiling of this vector.
    pub fn ceil(self) -> Vector3<S> {
        Vector3 {
            x: self.x.ceil(),
            y: self.y.ceil(),
            z: self.z.ceil(),
        }
    }

    /// Returns the element-wise rounded value of this vector.
    pub fn round(self) -> Vector3<S> {
        Vector3 {
            x: self.x.round(),
            y: self.y.round(),
            z: self.z.round(),
        }
    }

    /// Returns the element-wise truncated value of this vector.
    pub fn trunc(self) -> Vector3<S> {
        Vector3 {
            x: self.x.trunc(),
            y: self.y.trunc(),
            z: self.z.trunc(),
        }
    }

    /// Returns the element-wise fractional value of this vector.
    pub fn fract(self) -> Vector3<S> {
        Vector3 {
            x: self.x.fract(),
            y: self.y.fract(),
            z: self.z.fract(),
        }
    }

    /// Returns the element-wise absolute value of this vector.
    pub fn abs(self) -> Vector3<S> {
        Vector3 {
            x: self.x.abs(),
            y: self.y.abs(),
            z: self.z.abs(),
        }
    }

    /// Returns a vector with a permutation of the elements of this vector.
    pub fn permutation(self, dim_x: Dimension3, dim_y: Dimension3, dim_z: Dimension3) -> Vector3<S> {
        Vector3 {
            x: self[dim_x],
            y: self[dim_y],
            z: self[dim_z],
        }
    }
}

impl<S: Scalar> Index<Dimension3> for Vector3<S> {
    type Output = S;

    fn index(&self, index: Dimension3) -> &S {
        match index {
            Dimension3::X => &self.x,
            Dimension3::Y => &self.y,
            Dimension3::Z => &self.z,
        }
    }
}

impl<S: Scalar> IndexMut<Dimension3> for Vector3<S> {
    fn index_mut(&mut self, index: Dimension3) -> &mut S {
        match index {
            Dimension3::X => &mut self.x,
            Dimension3::Y => &mut self.y,
            Dimension3::Z => &mut self.z,
        }
    }
}

impl<S: Scalar> Zero for Vector3<S> {
    /// Returns the zero vector: (0, 0, 0).
    fn zero() -> Vector3<S> {
        Vector3::ZERO
    }

    /// Returns `true` if this vector is equal to the zero vector, `false` otherwise.
    fn is_zero(&self) -> bool {
        self.x.is_zero() && self.y.is_zero() && self.z.is_zero()
    }
}

impl<S: Scalar> ConstZero for Vector3<S> {
    /// The zero vector: (0, 0, 0).
    const ZERO: Vector3<S> = Vector3 { x: S::ZERO, y: S::ZERO, z: S::ZERO };
}

impl<S: Scalar> Add for Vector3<S> {
    type Output = Vector3<S>;

    fn add(self, other: Vector3<S>) -> Vector3<S> {
        Vector3 {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

impl<S: Scalar> AddAssign for Vector3<S> {
    fn add_assign(&mut self, other: Vector3<S>) {
        self.x += other.x;
        self.y += other.y;
        self.z += other.z;
    }
}

impl<S: Scalar> Sub for Vector3<S> {
    type Output = Vector3<S>;

    fn sub(self, other: Vector3<S>) -> Vector3<S> {
        Vector3 {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

impl<S: Scalar> SubAssign for Vector3<S> {
    fn sub_assign(&mut self, other: Vector3<S>) {
        self.x -= other.x;
        self.y -= other.y;
        self.z -= other.z;
    }
}

impl<S: Scalar> Neg for Vector3<S> {
    type Output = Vector3<S>;

    fn neg(self) -> Vector3<S> {
        Vector3 {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

impl<S: Scalar> Mul<S> for Vector3<S> {
    type Output = Vector3<S>;

    fn mul(self, value: S) -> Vector3<S> {
        Vector3 {
            x: self.x * value,
            y: self.y * value,
            z: self.z * value,
        }
    }
}

impl<S: Scalar> MulAssign<S> for Vector3<S> {
    fn mul_assign(&mut self, value: S) {
        self.x *= value;
        self.y *= value;
        self.z *= value;
    }
}

impl<S: Scalar> Div<S> for Vector3<S> {
    type Output = Vector3<S>;

    fn div(self, value: S) -> Vector3<S> {
        Vector3 {
            x: self.x / value,
            y: self.y / value,
            z: self.z / value,
        }
    }
}

impl<S: Scalar> DivAssign<S> for Vector3<S> {
    fn div_assign(&mut self, value: S) {
        self.x /= value;
        self.y /= value;
        self.z /= value;
    }
}

impl<S: Scalar> Rem<S> for Vector3<S> {
    type Output = Vector3<S>;

    fn rem(self, value: S) -> Vector3<S> {
        Vector3 {
            x: self.x % value,
            y: self.y % value,
            z: self.z % value,
        }
    }
}

impl<S: Scalar> RemAssign<S> for Vector3<S> {
    fn rem_assign(&mut self, value: S) {
        self.x %= value;
        self.y %= value;
        self.z %= value;
    }
}

impl<S: Scalar> MinMax for Vector3<S> {
    /// Returns the element-wise minimum of two vectors.
    fn min(self, other: Self) -> Self {
        Vector3 {
            x: self.x.min(other.x),
            y: self.y.min(other.y),
            z: self.z.min(other.z),
        }
    }

    /// Returns the element-wise maximum of two vectors.
    fn max(self, other: Self) -> Self {
        Vector3 {
            x: self.x.max(other.x),
            y: self.y.max(other.y),
            z: self.z.max(other.z),
        }
    }

    /// Returns the element-wise minimum and maximum of two vectors.
    fn min_max(self, other: Self) -> (Self, Self) {
        let (min_x, max_x) = self.x.min_max(other.x);
        let (min_y, max_y) = self.y.min_max(other.y);
        let (min_z, max_z) = self.z.min_max(other.z);

        (Vector3 { x: min_x, y: min_y, z: min_z }, Vector3 { x: max_x, y: max_y, z: max_z })
    }
}

impl<S: Scalar> Length<S> for Vector3<S> {
    /// Computes and returns the length of this vector.
    ///
    /// Note: If you only need to compare the lengths of vectors, use the methods of trait [RelativeLength] instead of calling this method.
    /// Computing the length of a vector involves a relatively expensive square root operation that can be avoided if you only need to
    /// compare the length of a vector to the length of another vector.
    fn length(self) -> S {
        self.dot(self).sqrt()
    }
}

impl<S: Scalar> RelativeLength<S> for Vector3<S> {
    fn cmp_length(self, other: Vector3<S>) -> Ordering {
        self.dot(self).total_cmp(&other.dot(other))
    }
}

impl<S: Scalar> DotProduct<S> for Vector3<S> {
    /// Computes and returns the dot product of this vector and another vector.
    fn dot(self, other: Vector3<S>) -> S {
        self.x * other.x + self.y * other.y + self.z * other.z
    }
}

impl<S: Scalar> DotProduct<S, Normal3<S>> for Vector3<S> {
    /// Computes and returns the dot product of this vector and a normal.
    fn dot(self, normal: Normal3<S>) -> S {
        self.x * normal.x + self.y * normal.y + self.z * normal.z
    }
}

impl<S: Scalar> From<Point3<S>> for Vector3<S> {
    fn from(point: Point3<S>) -> Vector3<S> {
        Vector3 {
            x: point.x,
            y: point.y,
            z: point.z,
        }
    }
}

impl<S: Scalar> From<Normal3<S>> for Vector3<S> {
    fn from(normal: Normal3<S>) -> Self {
        Vector3 {
            x: normal.x,
            y: normal.y,
            z: normal.z,
        }
    }
}

#[cfg(feature = "rand")]
impl<S: Scalar> Distribution<Vector3<S>> for Standard
where
    Standard: Distribution<(S, S, S)>,
{
    /// Generates a random vector with elements in the range `0..1` (uniformly distributed).
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Vector3<S> {
        let (x, y, z) = rng.gen();
        Vector3 { x, y, z }
    }
}
