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

use crate::{Dimension2, Dimension3, Distance, Length, MinMax, RelativeDistance, RelativeLength, Scalar, Vector2, Vector3};
use std::cmp::Ordering;
use std::ops::{Add, AddAssign, Index, IndexMut, Sub, SubAssign};

#[cfg(feature = "rand")]
use rand::{distributions::Standard, prelude::Distribution, Rng};

/// Point in 2D space.
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Point2<S: Scalar> {
    pub x: S,
    pub y: S,
}

/// Point in 3D space.
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Point3<S: Scalar> {
    pub x: S,
    pub y: S,
    pub z: S,
}

// ===== Point2 ================================================================================================================================================

impl<S: Scalar> Point2<S> {
    /// The origin: (0, 0).
    pub const ORIGIN: Point2<S> = Point2 { x: S::ZERO, y: S::ZERO };

    /// Creates and returns a new point.
    pub fn new(x: S, y: S) -> Point2<S> {
        Point2 { x, y }
    }

    /// Returns the origin: (0, 0).
    pub fn origin() -> Point2<S> {
        Point2::ORIGIN
    }

    /// Returns the dimension with the smallest distance from the origin of this point.
    ///
    /// # Example
    /// ```
    /// use vecmath::Point2;
    ///
    /// let v = Point2::new(-2.0, 1.0);
    /// println!("{:?}", v.min_dimension()); // prints: Y
    /// ```
    pub fn min_dimension(self) -> Dimension2 {
        let Point2 { x, y } = self.abs();
        if x <= y { Dimension2::X } else { Dimension2::Y }
    }

    /// Returns the dimension with the largest distance from the origin of this point.
    ///
    /// # Example
    /// ```
    /// use vecmath::Point2;
    ///
    /// let v = Point2::new(-2.0, 1.0);
    /// println!("{:?}", v.max_dimension()); // prints: X
    /// ```
    pub fn max_dimension(self) -> Dimension2 {
        let Point2 { x, y } = self.abs();
        if x > y { Dimension2::X } else { Dimension2::Y }
    }

    /// Returns the element-wise floor of this point.
    pub fn floor(self) -> Point2<S> {
        Point2 {
            x: self.x.floor(),
            y: self.y.floor(),
        }
    }

    /// Returns the element-wise ceiling of this point.
    pub fn ceil(self) -> Point2<S> {
        Point2 {
            x: self.x.ceil(),
            y: self.y.ceil(),
        }
    }

    /// Returns the element-wise rounded value of this point.
    pub fn round(self) -> Point2<S> {
        Point2 {
            x: self.x.round(),
            y: self.y.round(),
        }
    }

    /// Returns the element-wise truncated value of this point.
    pub fn trunc(self) -> Point2<S> {
        Point2 {
            x: self.x.trunc(),
            y: self.y.trunc(),
        }
    }

    /// Returns the element-wise fractional value of this point.
    pub fn fract(self) -> Point2<S> {
        Point2 {
            x: self.x.fract(),
            y: self.y.fract(),
        }
    }

    /// Returns the element-wise absolute value of this point.
    pub fn abs(self) -> Point2<S> {
        Point2 {
            x: self.x.abs(),
            y: self.y.abs(),
        }
    }

    /// Returns a point with a permutation of the elements of this point.
    pub fn permutation(self, dim_x: Dimension2, dim_y: Dimension2) -> Point2<S> {
        Point2 {
            x: self[dim_x],
            y: self[dim_y],
        }
    }
}

impl<S: Scalar> Index<Dimension2> for Point2<S> {
    type Output = S;

    fn index(&self, index: Dimension2) -> &S {
        match index {
            Dimension2::X => &self.x,
            Dimension2::Y => &self.y,
        }
    }
}

impl<S: Scalar> IndexMut<Dimension2> for Point2<S> {
    fn index_mut(&mut self, index: Dimension2) -> &mut S {
        match index {
            Dimension2::X => &mut self.x,
            Dimension2::Y => &mut self.y,
        }
    }
}

impl<S: Scalar> Add<Vector2<S>> for Point2<S> {
    type Output = Point2<S>;

    fn add(self, vector: Vector2<S>) -> Point2<S> {
        Point2 {
            x: self.x + vector.x,
            y: self.y + vector.y,
        }
    }
}

impl<S: Scalar> AddAssign<Vector2<S>> for Point2<S> {
    fn add_assign(&mut self, vector: Vector2<S>) {
        self.x += vector.x;
        self.y += vector.y;
    }
}

impl<S: Scalar> Sub<Vector2<S>> for Point2<S> {
    type Output = Point2<S>;

    fn sub(self, vector: Vector2<S>) -> Point2<S> {
        Point2 {
            x: self.x - vector.x,
            y: self.y - vector.y,
        }
    }
}

impl<S: Scalar> SubAssign<Vector2<S>> for Point2<S> {
    fn sub_assign(&mut self, vector: Vector2<S>) {
        self.x -= vector.x;
        self.y -= vector.y;
    }
}

impl<S: Scalar> Sub for Point2<S> {
    type Output = Vector2<S>;

    fn sub(self, other: Point2<S>) -> Vector2<S> {
        Vector2 {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }
}

impl<S: Scalar> MinMax for Point2<S> {
    /// Returns the element-wise minimum of two points.
    fn min(self, other: Self) -> Self {
        Point2 {
            x: self.x.min(other.x),
            y: self.y.min(other.y),
        }
    }

    /// Returns the element-wise maximum of two points.
    fn max(self, other: Self) -> Self {
        Point2 {
            x: self.x.max(other.x),
            y: self.y.max(other.y),
        }
    }

    /// Returns the element-wise minimum and maximum of two points.
    fn min_max(self, other: Self) -> (Self, Self) {
        let (min_x, max_x) = self.x.min_max(other.x);
        let (min_y, max_y) = self.y.min_max(other.y);

        (Point2 { x: min_x, y: min_y }, Point2 { x: max_x, y: max_y })
    }
}

impl<S: Scalar> Distance<S> for Point2<S> {
    /// Computes and returns the distance between this point and another point.
    ///
    /// Note: If you only need to compare distances between points, use the methods of trait [RelativeDistance] instead of calling this method.
    /// Computing the distance between points involves a relatively expensive square root operation that can be avoided if you only need to
    /// compare the distance of a point to two other points.
    fn distance(self, other: Point2<S>) -> S {
        (self - other).length()
    }
}

impl<S: Scalar> RelativeDistance<S> for Point2<S> {
    fn cmp_distance(self, first: Point2<S>, second: Point2<S>) -> Ordering {
        (self - first).cmp_length(self - second)
    }
}

impl<S: Scalar> From<Vector2<S>> for Point2<S> {
    fn from(vector: Vector2<S>) -> Self {
        Point2 {
            x: vector.x,
            y: vector.y,
        }
    }
}

#[cfg(feature = "rand")]
impl<S: Scalar> Distribution<Point2<S>> for Standard
where
    Standard: Distribution<(S, S)>,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Point2<S> {
        let (x, y) = rng.gen();
        Point2 { x, y }
    }
}

// ===== Point3 ================================================================================================================================================

impl<S: Scalar> Point3<S> {
    /// The origin: (0, 0, 0).
    pub const ORIGIN: Point3<S> = Point3 { x: S::ZERO, y: S::ZERO, z: S::ZERO };

    /// Creates and returns a new point.
    pub fn new(x: S, y: S, z: S) -> Point3<S> {
        Point3 { x, y, z }
    }

    /// Returns the origin: (0, 0, 0).
    pub fn origin() -> Point3<S> {
        Point3::ORIGIN
    }

    /// Returns the dimension with the smallest distance from the origin of this point.
    ///
    /// # Example
    /// ```
    /// use vecmath::Point3;
    ///
    /// let v = Point3::new(-2.0, 1.0, 1.5);
    /// println!("{:?}", v.min_dimension()); // prints: Y
    /// ```
    pub fn min_dimension(self) -> Dimension3 {
        let Point3 { x, y, z } = self.abs();
        if x <= y && x <= z { Dimension3::X } else if y <= z { Dimension3::Y } else { Dimension3::Z }
    }

    /// Returns the dimension with the largest distance from the origin of this point.
    ///
    /// # Example
    /// ```
    /// use vecmath::Point3;
    ///
    /// let v = Point3::new(-2.0, 1.0, 1.5);
    /// println!("{:?}", v.max_dimension()); // prints: X
    /// ```
    pub fn max_dimension(self) -> Dimension3 {
        let Point3 { x, y, z } = self.abs();
        if x > y && x > z { Dimension3::X } else if y > z { Dimension3::Y } else { Dimension3::Z }
    }

    /// Returns the element-wise floor of this point.
    pub fn floor(self) -> Point3<S> {
        Point3 {
            x: self.x.floor(),
            y: self.y.floor(),
            z: self.z.floor(),
        }
    }

    /// Returns the element-wise ceiling of this point.
    pub fn ceil(self) -> Point3<S> {
        Point3 {
            x: self.x.ceil(),
            y: self.y.ceil(),
            z: self.z.ceil(),
        }
    }

    /// Returns the element-wise rounded value of this point.
    pub fn round(self) -> Point3<S> {
        Point3 {
            x: self.x.round(),
            y: self.y.round(),
            z: self.z.round(),
        }
    }

    /// Returns the element-wise truncated value of this point.
    pub fn trunc(self) -> Point3<S> {
        Point3 {
            x: self.x.trunc(),
            y: self.y.trunc(),
            z: self.z.trunc(),
        }
    }

    /// Returns the element-wise fractional value of this point.
    pub fn fract(self) -> Point3<S> {
        Point3 {
            x: self.x.fract(),
            y: self.y.fract(),
            z: self.z.fract(),
        }
    }

    /// Returns the element-wise absolute value of this point.
    pub fn abs(self) -> Point3<S> {
        Point3 {
            x: self.x.abs(),
            y: self.y.abs(),
            z: self.z.abs(),
        }
    }

    /// Returns a point with a permutation of the elements of this point.
    pub fn permutation(self, dim_x: Dimension3, dim_y: Dimension3, dim_z: Dimension3) -> Point3<S> {
        Point3 {
            x: self[dim_x],
            y: self[dim_y],
            z: self[dim_z],
        }
    }
}

impl<S: Scalar> Index<Dimension3> for Point3<S> {
    type Output = S;

    fn index(&self, index: Dimension3) -> &S {
        match index {
            Dimension3::X => &self.x,
            Dimension3::Y => &self.y,
            Dimension3::Z => &self.z,
        }
    }
}

impl<S: Scalar> IndexMut<Dimension3> for Point3<S> {
    fn index_mut(&mut self, index: Dimension3) -> &mut S {
        match index {
            Dimension3::X => &mut self.x,
            Dimension3::Y => &mut self.y,
            Dimension3::Z => &mut self.z,
        }
    }
}

impl<S: Scalar> Add<Vector3<S>> for Point3<S> {
    type Output = Point3<S>;

    fn add(self, vector: Vector3<S>) -> Point3<S> {
        Point3 {
            x: self.x + vector.x,
            y: self.y + vector.y,
            z: self.z + vector.z,
        }
    }
}

impl<S: Scalar> AddAssign<Vector3<S>> for Point3<S> {
    fn add_assign(&mut self, vector: Vector3<S>) {
        self.x += vector.x;
        self.y += vector.y;
        self.z += vector.z;
    }
}

impl<S: Scalar> Sub<Vector3<S>> for Point3<S> {
    type Output = Point3<S>;

    fn sub(self, vector: Vector3<S>) -> Point3<S> {
        Point3 {
            x: self.x - vector.x,
            y: self.y - vector.y,
            z: self.z - vector.z,
        }
    }
}

impl<S: Scalar> SubAssign<Vector3<S>> for Point3<S> {
    fn sub_assign(&mut self, vector: Vector3<S>) {
        self.x -= vector.x;
        self.y -= vector.y;
        self.z -= vector.z;
    }
}

impl<S: Scalar> Sub for Point3<S> {
    type Output = Vector3<S>;

    fn sub(self, other: Point3<S>) -> Vector3<S> {
        Vector3 {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

impl<S: Scalar> MinMax for Point3<S> {
    /// Returns the element-wise minimum of two points.
    fn min(self, other: Self) -> Self {
        Point3 {
            x: self.x.min(other.x),
            y: self.y.min(other.y),
            z: self.z.min(other.z),
        }
    }

    /// Returns the element-wise maximum of two points.
    fn max(self, other: Self) -> Self {
        Point3 {
            x: self.x.max(other.x),
            y: self.y.max(other.y),
            z: self.z.max(other.z),
        }
    }

    /// Returns the element-wise minimum and maximum of two points.
    fn min_max(self, other: Self) -> (Self, Self) {
        let (min_x, max_x) = self.x.min_max(other.x);
        let (min_y, max_y) = self.y.min_max(other.y);
        let (min_z, max_z) = self.z.min_max(other.z);

        (Point3 { x: min_x, y: min_y, z: min_z }, Point3 { x: max_x, y: max_y, z: max_z })
    }
}

impl<S: Scalar> Distance<S> for Point3<S> {
    /// Computes and returns the distance between this point and another point.
    ///
    /// Note: If you only need to compare distances between points, use the methods of trait [RelativeDistance] instead of calling this method.
    /// Computing the distance between points involves a relatively expensive square root operation that can be avoided if you only need to
    /// compare the distance of a point to two other points.
    fn distance(self, other: Point3<S>) -> S {
        (self - other).length()
    }
}

impl<S: Scalar> RelativeDistance<S> for Point3<S> {
    fn cmp_distance(self, first: Point3<S>, second: Point3<S>) -> Ordering {
        (self - first).cmp_length(self - second)
    }
}

impl<S: Scalar> From<Vector3<S>> for Point3<S> {
    fn from(vector: Vector3<S>) -> Point3<S> {
        Point3 {
            x: vector.x,
            y: vector.y,
            z: vector.z,
        }
    }
}

#[cfg(feature = "rand")]
impl<S: Scalar> Distribution<Point3<S>> for Standard
where
    Standard: Distribution<(S, S, S)>,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Point3<S> {
        let (x, y, z) = rng.gen();
        Point3 { x, y, z }
    }
}
