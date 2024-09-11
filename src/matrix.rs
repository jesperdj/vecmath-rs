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

use crate::{Angle, Normal3, Point2, Point3, Scalar, Vector2, Vector3};
use std::array;
use std::error::Error;
use std::fmt::{Debug, Display, Formatter};
use std::ops::{Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign};

/// Matrix with 3 rows and 3 columns for transformations in 2D space.
#[derive(Clone, PartialEq, Debug)]
pub struct Matrix3x3<S: Scalar> {
    elements: [S; 9],
}

/// Matrix with 4 rows and 4 columns for transformations in 3D space.
#[derive(Clone, PartialEq, Debug)]
pub struct Matrix4x4<S: Scalar> {
    elements: [S; 16],
}

/// Error returned when computing the inverse of a singular matrix is attempted.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct NonInvertibleMatrixError;

// ===== Matrix3x3 =============================================================================================================================================

impl<S: Scalar> Matrix3x3<S> {
    /// The identity matrix.
    pub const IDENTITY: Matrix3x3<S> = Matrix3x3 {
        elements: [
            S::ONE, S::ZERO, S::ZERO,
            S::ZERO, S::ONE, S::ZERO,
            S::ZERO, S::ZERO, S::ONE,
        ]
    };

    /// Returns the identity matrix.
    pub fn identity() -> Matrix3x3<S> {
        Matrix3x3::IDENTITY
    }

    /// Returns `true` if this matrix is equal to the identity matrix, `false` otherwise.
    pub fn is_identity(&self) -> bool {
        *self == Matrix3x3::IDENTITY
    }

    /// Returns a transformation matrix for translation in 2D space.
    pub fn translate(vector: Vector2<S>) -> Matrix3x3<S> {
        Matrix3x3 {
            elements: [
                S::ONE, S::ZERO, vector.x,
                S::ZERO, S::ONE, vector.y,
                S::ZERO, S::ZERO, S::ONE,
            ]
        }
    }

    /// Returns a transformation matrix for rotating around the origin in 2D space.
    pub fn rotate(angle: Angle<S>) -> Matrix3x3<S> {
        let (sin, cos) = angle.radians().sin_cos();

        Matrix3x3 {
            elements: [
                cos, -sin, S::ZERO,
                sin, cos, S::ZERO,
                S::ZERO, S::ZERO, S::ONE,
            ]
        }
    }

    /// Returns a transformation matrix for scaling in 2D space in the X and Y dimensions.
    pub fn scale(factor_x: S, factor_y: S) -> Matrix3x3<S> {
        Matrix3x3 {
            elements: [
                factor_x, S::ZERO, S::ZERO,
                S::ZERO, factor_y, S::ZERO,
                S::ZERO, S::ZERO, S::ONE,
            ]
        }
    }

    /// Returns a transformation matrix for scaling uniformly in all dimensions in 2D space.
    pub fn scale_uniform(factor: S) -> Matrix3x3<S> {
        Matrix3x3 {
            elements: [
                factor, S::ZERO, S::ZERO,
                S::ZERO, factor, S::ZERO,
                S::ZERO, S::ZERO, S::ONE,
            ]
        }
    }

    /// Creates a matrix with the specified elements.
    pub fn with_elements(elements: [S; 9]) -> Matrix3x3<S> {
        Matrix3x3 { elements }
    }

    /// Returns an element at a row and column of the matrix.
    ///
    /// Both `row` and `col` must be in the range `0..=2`.
    pub fn get(&self, row: u32, col: u32) -> S {
        self.elements[Self::linear_index(row, col)]
    }

    /// Returns a mutable reference to an element at a row and column of the matrix.
    ///
    /// Both `row` and `col` must be in the range `0..=2`.
    pub fn get_mut(&mut self, row: u32, col: u32) -> &mut S {
        &mut self.elements[Self::linear_index(row, col)]
    }

    /// Sets the value of an element at a row and column of the matrix.
    ///
    /// Both `row` and `col` must be in the range `0..=2`.
    pub fn set(&mut self, row: u32, col: u32, value: S) {
        self.elements[Self::linear_index(row, col)] = value;
    }

    fn linear_index(row: u32, col: u32) -> usize {
        debug_assert!(row < 3, "Invalid row index: {}", row);
        debug_assert!(col < 3, "Invalid column index: {}", col);
        (row * 3 + col) as usize
    }

    /// Computes and returns the transpose of this matrix.
    pub fn transposed(&self) -> Matrix3x3<S> {
        Matrix3x3 {
            elements: [
                self.elements[0], self.elements[3], self.elements[6],
                self.elements[1], self.elements[4], self.elements[7],
                self.elements[2], self.elements[5], self.elements[8],
            ]
        }
    }

    /// Transposes this matrix.
    pub fn transpose(&mut self) {
        *self = self.transposed();
    }

    /// Computes and returns the inverse of this matrix.
    ///
    /// If this matrix is singular, a `NonInvertibleMatrixError` is returned.
    pub fn inverted(&self) -> Result<Matrix3x3<S>, NonInvertibleMatrixError> {
        let det = self.elements[0] * self.elements[4] * self.elements[8]
            + self.elements[1] * self.elements[5] * self.elements[6]
            + self.elements[2] * self.elements[3] * self.elements[7]
            - self.elements[2] * self.elements[4] * self.elements[6]
            - self.elements[1] * self.elements[3] * self.elements[8]
            - self.elements[0] * self.elements[5] * self.elements[7];

        if !det.is_zero() {
            Ok(Matrix3x3 {
                elements: [
                    (self.elements[4] * self.elements[8] - self.elements[5] * self.elements[7]) / det,
                    (self.elements[2] * self.elements[7] - self.elements[1] * self.elements[8]) / det,
                    (self.elements[1] * self.elements[5] - self.elements[2] * self.elements[4]) / det,
                    (self.elements[5] * self.elements[6] - self.elements[3] * self.elements[8]) / det,
                    (self.elements[0] * self.elements[8] - self.elements[2] * self.elements[6]) / det,
                    (self.elements[2] * self.elements[3] - self.elements[0] * self.elements[5]) / det,
                    (self.elements[3] * self.elements[7] - self.elements[4] * self.elements[6]) / det,
                    (self.elements[1] * self.elements[6] - self.elements[0] * self.elements[7]) / det,
                    (self.elements[0] * self.elements[4] - self.elements[1] * self.elements[3]) / det,
                ]
            })
        } else {
            Err(NonInvertibleMatrixError)
        }
    }
}

impl<S: Scalar> Mul for &Matrix3x3<S> {
    type Output = Matrix3x3<S>;

    fn mul(self, other: &Matrix3x3<S>) -> Matrix3x3<S> {
        Matrix3x3 {
            elements: [
                self.elements[0] * other.elements[0] + self.elements[1] * other.elements[3] + self.elements[2] * other.elements[6],
                self.elements[0] * other.elements[1] + self.elements[1] * other.elements[4] + self.elements[2] * other.elements[7],
                self.elements[0] * other.elements[2] + self.elements[1] * other.elements[5] + self.elements[2] * other.elements[8],
                self.elements[3] * other.elements[0] + self.elements[4] * other.elements[3] + self.elements[5] * other.elements[6],
                self.elements[3] * other.elements[1] + self.elements[4] * other.elements[4] + self.elements[5] * other.elements[7],
                self.elements[3] * other.elements[2] + self.elements[4] * other.elements[5] + self.elements[5] * other.elements[8],
                self.elements[6] * other.elements[0] + self.elements[7] * other.elements[3] + self.elements[8] * other.elements[6],
                self.elements[6] * other.elements[1] + self.elements[7] * other.elements[4] + self.elements[8] * other.elements[7],
                self.elements[6] * other.elements[2] + self.elements[7] * other.elements[5] + self.elements[8] * other.elements[8],
            ]
        }
    }
}

impl<S: Scalar> MulAssign<&Matrix3x3<S>> for Matrix3x3<S> {
    fn mul_assign(&mut self, other: &Matrix3x3<S>) {
        *self = &*self * other;
    }
}

impl<S: Scalar> Neg for &Matrix3x3<S> {
    type Output = Matrix3x3<S>;

    fn neg(self) -> Matrix3x3<S> {
        Matrix3x3 { elements: array::from_fn(|i| -self.elements[i]) }
    }
}

impl<S: Scalar> Mul<S> for &Matrix3x3<S> {
    type Output = Matrix3x3<S>;

    fn mul(self, value: S) -> Matrix3x3<S> {
        Matrix3x3 { elements: array::from_fn(|i| self.elements[i] * value) }
    }
}

impl<S: Scalar> MulAssign<S> for Matrix3x3<S> {
    fn mul_assign(&mut self, value: S) {
        for e in &mut self.elements { *e *= value; }
    }
}

impl<S: Scalar> Div<S> for &Matrix3x3<S> {
    type Output = Matrix3x3<S>;

    fn div(self, value: S) -> Matrix3x3<S> {
        Matrix3x3 { elements: array::from_fn(|i| self.elements[i] / value) }
    }
}

impl<S: Scalar> DivAssign<S> for Matrix3x3<S> {
    fn div_assign(&mut self, value: S) {
        for e in &mut self.elements { *e /= value; }
    }
}

impl<S: Scalar> Rem<S> for &Matrix3x3<S> {
    type Output = Matrix3x3<S>;

    fn rem(self, value: S) -> Matrix3x3<S> {
        Matrix3x3 { elements: array::from_fn(|i| self.elements[i] % value) }
    }
}

impl<S: Scalar> RemAssign<S> for Matrix3x3<S> {
    fn rem_assign(&mut self, value: S) {
        for e in &mut self.elements { *e %= value; }
    }
}

impl<S: Scalar> Mul<Vector2<S>> for &Matrix3x3<S> {
    type Output = Vector2<S>;

    fn mul(self, vector: Vector2<S>) -> Vector2<S> {
        Vector2 {
            x: self.elements[0] * vector.x + self.elements[1] * vector.y,
            y: self.elements[3] * vector.x + self.elements[4] * vector.y,
        }
    }
}

impl<S: Scalar> Mul<Point2<S>> for &Matrix3x3<S> {
    type Output = Point2<S>;

    fn mul(self, point: Point2<S>) -> Point2<S> {
        let w = self.elements[6] * point.x + self.elements[7] * point.y + self.elements[8];

        Point2 {
            x: (self.elements[0] * point.x + self.elements[1] * point.y + self.elements[2]) / w,
            y: (self.elements[3] * point.x + self.elements[4] * point.y + self.elements[5]) / w,
        }
    }
}

// ===== Matrix4x4 =============================================================================================================================================

impl<S: Scalar> Matrix4x4<S> {
    /// The identity matrix.
    pub const IDENTITY: Matrix4x4<S> = Matrix4x4 {
        elements: [
            S::ONE, S::ZERO, S::ZERO, S::ZERO,
            S::ZERO, S::ONE, S::ZERO, S::ZERO,
            S::ZERO, S::ZERO, S::ONE, S::ZERO,
            S::ZERO, S::ZERO, S::ZERO, S::ONE,
        ]
    };

    /// Returns the identity matrix.
    pub fn identity() -> Matrix4x4<S> {
        Matrix4x4::IDENTITY
    }

    /// Returns `true` if this matrix is equal to the identity matrix, `false` otherwise.
    pub fn is_identity(&self) -> bool {
        *self == Matrix4x4::IDENTITY
    }

    /// Returns a transformation matrix for translation in 3D space.
    pub fn translate(vector: Vector3<S>) -> Matrix4x4<S> {
        Matrix4x4 {
            elements: [
                S::ONE, S::ZERO, S::ZERO, vector.x,
                S::ZERO, S::ONE, S::ZERO, vector.y,
                S::ZERO, S::ZERO, S::ONE, vector.z,
                S::ZERO, S::ZERO, S::ZERO, S::ONE,
            ]
        }
    }

    /// Returns a transformation matrix for rotating around the X axis in 3D space.
    pub fn rotate_x(angle: Angle<S>) -> Matrix4x4<S> {
        let (sin, cos) = angle.radians().sin_cos();

        Matrix4x4 {
            elements: [
                S::ONE, S::ZERO, S::ZERO, S::ZERO,
                S::ZERO, cos, -sin, S::ZERO,
                S::ZERO, sin, cos, S::ZERO,
                S::ZERO, S::ZERO, S::ZERO, S::ONE,
            ]
        }
    }

    /// Returns a transformation matrix for rotating around the Y axis in 3D space.
    pub fn rotate_y(angle: Angle<S>) -> Matrix4x4<S> {
        let (sin, cos) = angle.radians().sin_cos();

        Matrix4x4 {
            elements: [
                cos, S::ZERO, sin, S::ZERO,
                S::ZERO, S::ONE, S::ZERO, S::ZERO,
                -sin, S::ZERO, cos, S::ZERO,
                S::ZERO, S::ZERO, S::ZERO, S::ONE,
            ]
        }
    }

    /// Returns a transformation matrix for rotating around the Z axis in 3D space.
    pub fn rotate_z(angle: Angle<S>) -> Matrix4x4<S> {
        let (sin, cos) = angle.radians().sin_cos();

        Matrix4x4 {
            elements: [
                cos, -sin, S::ZERO, S::ZERO,
                sin, cos, S::ZERO, S::ZERO,
                S::ZERO, S::ZERO, S::ONE, S::ZERO,
                S::ZERO, S::ZERO, S::ZERO, S::ONE,
            ]
        }
    }

    /// Returns a transformation matrix for rotating around an axis in 3D space.
    pub fn rotate_axis(axis: Vector3<S>, angle: Angle<S>) -> Matrix4x4<S> {
        let a = axis.normalized();
        let (sin, cos) = angle.radians().sin_cos();
        let cc = S::ONE - cos;

        let (t1, t2, t3) = (a.x * a.y * cc, a.x * a.z * cc, a.y * a.z * cc);
        let (u1, u2, u3) = (a.x * sin, a.y * sin, a.z * sin);

        Matrix4x4 {
            elements: [
                a.x * a.x * cc + cos, t1 - u3, t2 + u2, S::ZERO,
                t1 + u3, a.y * a.y * cc + cos, t3 - u1, S::ZERO,
                t2 - u2, t3 + u1, a.z * a.z * cc + cos, S::ZERO,
                S::ZERO, S::ZERO, S::ZERO, S::ONE,
            ]
        }
    }

    /// Returns a transformation matrix for scaling in 3D space in the X, Y and Z dimensions.
    pub fn scale(factor_x: S, factor_y: S, factor_z: S) -> Matrix4x4<S> {
        Matrix4x4 {
            elements: [
                factor_x, S::ZERO, S::ZERO, S::ZERO,
                S::ZERO, factor_y, S::ZERO, S::ZERO,
                S::ZERO, S::ZERO, factor_z, S::ZERO,
                S::ZERO, S::ZERO, S::ZERO, S::ONE,
            ]
        }
    }

    /// Returns a transformation matrix for scaling uniformly in all dimensions in 3D space.
    pub fn scale_uniform(factor: S) -> Matrix4x4<S> {
        Matrix4x4 {
            elements: [
                factor, S::ZERO, S::ZERO, S::ZERO,
                S::ZERO, factor, S::ZERO, S::ZERO,
                S::ZERO, S::ZERO, factor, S::ZERO,
                S::ZERO, S::ZERO, S::ZERO, S::ONE,
            ]
        }
    }

    /// Returns a "look at" transformation matrix.
    ///
    /// See [Transform3::look_at()] and [Physically Based Rendering: The Look-at Transformation](https://pbr-book.org/4ed/Geometry_and_Transformations/Transformations#TheLook-atTransformation).
    pub fn look_at(from: Point3<S>, target: Point3<S>, up: Vector3<S>) -> Matrix4x4<S> {
        let direction = (target - from).normalized();
        let right = up.normalized().cross(direction).normalized();
        let new_up = direction.cross(right);

        Matrix4x4 {
            elements: [
                right.x, new_up.x, direction.x, from.x,
                right.y, new_up.y, direction.y, from.y,
                right.z, new_up.z, direction.z, from.z,
                S::ZERO, S::ZERO, S::ZERO, S::ONE,
            ]
        }
    }

    /// Creates a matrix with the specified elements.
    pub fn with_elements(elements: [S; 16]) -> Matrix4x4<S> {
        Matrix4x4 { elements }
    }

    /// Returns an element at a row and column of the matrix.
    ///
    /// Both `row` and `col` must be in the range `0..=3`.
    pub fn get(&self, row: u32, col: u32) -> S {
        self.elements[Self::linear_index(row, col)]
    }

    /// Returns a mutable reference to an element at a row and column of the matrix.
    ///
    /// Both `row` and `col` must be in the range `0..=3`.
    pub fn get_mut(&mut self, row: u32, col: u32) -> &mut S {
        &mut self.elements[Self::linear_index(row, col)]
    }

    /// Sets the value of an element at a row and column of the matrix.
    ///
    /// Both `row` and `col` must be in the range `0..=3`.
    pub fn set(&mut self, row: u32, col: u32, value: S) {
        self.elements[Self::linear_index(row, col)] = value;
    }

    fn linear_index(row: u32, col: u32) -> usize {
        debug_assert!(row < 4, "Invalid row index: {}", row);
        debug_assert!(col < 4, "Invalid column index: {}", col);
        (row * 4 + col) as usize
    }

    /// Computes and returns the transpose of this matrix.
    pub fn transposed(&self) -> Matrix4x4<S> {
        Matrix4x4 {
            elements: [
                self.elements[0], self.elements[4], self.elements[8], self.elements[12],
                self.elements[1], self.elements[5], self.elements[9], self.elements[13],
                self.elements[2], self.elements[6], self.elements[10], self.elements[14],
                self.elements[3], self.elements[7], self.elements[11], self.elements[15],
            ]
        }
    }

    /// Transposes this matrix.
    pub fn transpose(&mut self) {
        *self = self.transposed();
    }

    /// Computes and returns the inverse of this matrix.
    ///
    /// If this matrix is singular, a `NonInvertibleMatrixError` is returned.
    pub fn inverted(&self) -> Result<Matrix4x4<S>, NonInvertibleMatrixError> {
        let cofactor = |i, j| {
            let sub = |row, col| self.get(if row < i { row } else { row + 1 }, if col < j { col } else { col + 1 });

            let sign = if (i + j) % 2 == 0 { S::ONE } else { -S::ONE };

            sign * (sub(0, 0) * sub(1, 1) * sub(2, 2)
                + sub(0, 1) * sub(1, 2) * sub(2, 0)
                + sub(0, 2) * sub(1, 0) * sub(2, 1)
                - sub(0, 0) * sub(1, 2) * sub(2, 1)
                - sub(0, 1) * sub(1, 0) * sub(2, 2)
                - sub(0, 2) * sub(1, 1) * sub(2, 0))
        };

        let adjugate = Matrix4x4 {
            elements: [
                cofactor(0, 0), cofactor(1, 0), cofactor(2, 0), cofactor(3, 0),
                cofactor(0, 1), cofactor(1, 1), cofactor(2, 1), cofactor(3, 1),
                cofactor(0, 2), cofactor(1, 2), cofactor(2, 2), cofactor(3, 2),
                cofactor(0, 3), cofactor(1, 3), cofactor(2, 3), cofactor(3, 3),
            ]
        };

        let det = self.elements[0] * adjugate.elements[0]
            + self.elements[1] * adjugate.elements[4]
            + self.elements[2] * adjugate.elements[8]
            + self.elements[3] * adjugate.elements[12];

        if !det.is_zero() {
            Ok(&adjugate / det)
        } else {
            Err(NonInvertibleMatrixError)
        }
    }
}

impl<S: Scalar> Mul for &Matrix4x4<S> {
    type Output = Matrix4x4<S>;

    fn mul(self, other: &Matrix4x4<S>) -> Matrix4x4<S> {
        Matrix4x4 {
            elements: [
                self.elements[0] * other.elements[0] + self.elements[1] * other.elements[4] + self.elements[2] * other.elements[8] + self.elements[3] * other.elements[12],
                self.elements[0] * other.elements[1] + self.elements[1] * other.elements[5] + self.elements[2] * other.elements[9] + self.elements[3] * other.elements[13],
                self.elements[0] * other.elements[2] + self.elements[1] * other.elements[6] + self.elements[2] * other.elements[10] + self.elements[3] * other.elements[14],
                self.elements[0] * other.elements[3] + self.elements[1] * other.elements[7] + self.elements[2] * other.elements[11] + self.elements[3] * other.elements[15],
                self.elements[4] * other.elements[0] + self.elements[5] * other.elements[4] + self.elements[6] * other.elements[8] + self.elements[7] * other.elements[12],
                self.elements[4] * other.elements[1] + self.elements[5] * other.elements[5] + self.elements[6] * other.elements[9] + self.elements[7] * other.elements[13],
                self.elements[4] * other.elements[2] + self.elements[5] * other.elements[6] + self.elements[6] * other.elements[10] + self.elements[7] * other.elements[14],
                self.elements[4] * other.elements[3] + self.elements[5] * other.elements[7] + self.elements[6] * other.elements[11] + self.elements[7] * other.elements[15],
                self.elements[8] * other.elements[0] + self.elements[9] * other.elements[4] + self.elements[10] * other.elements[8] + self.elements[11] * other.elements[12],
                self.elements[8] * other.elements[1] + self.elements[9] * other.elements[5] + self.elements[10] * other.elements[9] + self.elements[11] * other.elements[13],
                self.elements[8] * other.elements[2] + self.elements[9] * other.elements[6] + self.elements[10] * other.elements[10] + self.elements[11] * other.elements[14],
                self.elements[8] * other.elements[3] + self.elements[9] * other.elements[7] + self.elements[10] * other.elements[11] + self.elements[11] * other.elements[15],
                self.elements[12] * other.elements[0] + self.elements[13] * other.elements[4] + self.elements[14] * other.elements[8] + self.elements[15] * other.elements[12],
                self.elements[12] * other.elements[1] + self.elements[13] * other.elements[5] + self.elements[14] * other.elements[9] + self.elements[15] * other.elements[13],
                self.elements[12] * other.elements[2] + self.elements[13] * other.elements[6] + self.elements[14] * other.elements[10] + self.elements[15] * other.elements[14],
                self.elements[12] * other.elements[3] + self.elements[13] * other.elements[7] + self.elements[14] * other.elements[11] + self.elements[15] * other.elements[15],
            ]
        }
    }
}

impl<S: Scalar> MulAssign<&Matrix4x4<S>> for Matrix4x4<S> {
    fn mul_assign(&mut self, other: &Matrix4x4<S>) {
        *self = &*self * other;
    }
}

impl<S: Scalar> Neg for &Matrix4x4<S> {
    type Output = Matrix4x4<S>;

    fn neg(self) -> Matrix4x4<S> {
        Matrix4x4 { elements: array::from_fn(|i| -self.elements[i]) }
    }
}

impl<S: Scalar> Mul<S> for &Matrix4x4<S> {
    type Output = Matrix4x4<S>;

    fn mul(self, value: S) -> Matrix4x4<S> {
        Matrix4x4 { elements: array::from_fn(|i| self.elements[i] * value) }
    }
}

impl<S: Scalar> MulAssign<S> for Matrix4x4<S> {
    fn mul_assign(&mut self, value: S) {
        for e in &mut self.elements { *e *= value; }
    }
}

impl<S: Scalar> Div<S> for &Matrix4x4<S> {
    type Output = Matrix4x4<S>;

    fn div(self, value: S) -> Matrix4x4<S> {
        Matrix4x4 { elements: array::from_fn(|i| self.elements[i] / value) }
    }
}

impl<S: Scalar> DivAssign<S> for Matrix4x4<S> {
    fn div_assign(&mut self, value: S) {
        for e in &mut self.elements { *e /= value; }
    }
}

impl<S: Scalar> Rem<S> for &Matrix4x4<S> {
    type Output = Matrix4x4<S>;

    fn rem(self, value: S) -> Matrix4x4<S> {
        Matrix4x4 { elements: array::from_fn(|i| self.elements[i] % value) }
    }
}

impl<S: Scalar> RemAssign<S> for Matrix4x4<S> {
    fn rem_assign(&mut self, value: S) {
        for e in &mut self.elements { *e %= value; }
    }
}

impl<S: Scalar> Mul<Vector3<S>> for &Matrix4x4<S> {
    type Output = Vector3<S>;

    fn mul(self, vector: Vector3<S>) -> Vector3<S> {
        Vector3 {
            x: self.elements[0] * vector.x + self.elements[1] * vector.y + self.elements[2] * vector.z,
            y: self.elements[4] * vector.x + self.elements[5] * vector.y + self.elements[6] * vector.z,
            z: self.elements[8] * vector.x + self.elements[9] * vector.y + self.elements[10] * vector.z,
        }
    }
}

impl<S: Scalar> Mul<Point3<S>> for &Matrix4x4<S> {
    type Output = Point3<S>;

    fn mul(self, point: Point3<S>) -> Point3<S> {
        let w = self.elements[12] * point.x + self.elements[13] * point.y + self.elements[14] * point.z + self.elements[15];

        Point3 {
            x: (self.elements[0] * point.x + self.elements[1] * point.y + self.elements[2] * point.z + self.elements[3]) / w,
            y: (self.elements[4] * point.x + self.elements[5] * point.y + self.elements[6] * point.z + self.elements[7]) / w,
            z: (self.elements[8] * point.x + self.elements[9] * point.y + self.elements[10] * point.z + self.elements[11]) / w,
        }
    }
}

impl<S: Scalar> Mul<Normal3<S>> for &Matrix4x4<S> {
    type Output = Normal3<S>;

    fn mul(self, normal: Normal3<S>) -> Normal3<S> {
        Normal3 {
            x: self.elements[0] * normal.x + self.elements[1] * normal.y + self.elements[2] * normal.z,
            y: self.elements[4] * normal.x + self.elements[5] * normal.y + self.elements[6] * normal.z,
            z: self.elements[8] * normal.x + self.elements[9] * normal.y + self.elements[10] * normal.z,
        }
    }
}

// ===== NonInvertibleMatrixError ==============================================================================================================================

impl Error for NonInvertibleMatrixError {}

impl Display for NonInvertibleMatrixError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Matrix is not invertible")
    }
}
