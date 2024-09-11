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

use crate::{Angle, BoundingBox2, BoundingBox3, Length, Matrix3x3, Matrix4x4, NonInvertibleMatrixError, Normal3, Point2, Point3, Ray2, Ray3, Scalar, Vector2, Vector3};

/// Trait for types that can transform a value of type `T`.
pub trait Transform<T> {
    /// The output type that results from transforming a value of type `T`.
    type Output;

    /// Transforms a value of type `T`.
    fn transform(&self, value: T) -> Self::Output;

    /// Transforms a value of type `T` using the inverse transformation.
    fn inverse_transform(&self, value: T) -> Self::Output;
}

/// Transform for transformations in 2D space.
#[derive(Clone, PartialEq, Debug)]
pub struct Transform2<S: Scalar> {
    forward: Matrix3x3<S>,
    inverse: Matrix3x3<S>,
}

/// Transform for transformations in 3D space.
#[derive(Clone, PartialEq, Debug)]
pub struct Transform3<S: Scalar> {
    forward: Matrix4x4<S>,
    inverse: Matrix4x4<S>,
}

// ===== Transform2 ============================================================================================================================================

impl<S: Scalar> Transform2<S> {
    /// The identity transform.
    pub const IDENTITY: Transform2<S> = Transform2 {
        forward: Matrix3x3::IDENTITY,
        inverse: Matrix3x3::IDENTITY,
    };

    /// Returns the identity transform.
    pub fn identity() -> Transform2<S> {
        Transform2::IDENTITY
    }

    /// Returns `true` if this transform is equal to the identity transform, `false` otherwise.
    pub fn is_identity(&self) -> bool {
        *self == Transform2::IDENTITY
    }

    /// Returns a transform for translation in 2D space.
    pub fn translate(vector: Vector2<S>) -> Transform2<S> {
        Transform2 {
            forward: Matrix3x3::translate(vector),
            inverse: Matrix3x3::translate(-vector),
        }
    }

    /// Returns a transform for rotating around the origin in 2D space.
    pub fn rotate(angle: Angle<S>) -> Transform2<S> {
        Transform2 {
            forward: Matrix3x3::rotate(angle),
            inverse: Matrix3x3::rotate(-angle),
        }
    }

    /// Returns a transform for scaling in 2D space in the X and Y directions.
    pub fn scale(factor_x: S, factor_y: S) -> Transform2<S> {
        Transform2 {
            forward: Matrix3x3::scale(factor_x, factor_y),
            inverse: Matrix3x3::scale(factor_x.recip(), factor_y.recip()),
        }
    }

    /// Returns a transform for scaling uniformly in 2D space.
    pub fn scale_uniform(factor: S) -> Transform2<S> {
        Transform2 {
            forward: Matrix3x3::scale_uniform(factor),
            inverse: Matrix3x3::scale_uniform(factor.recip()),
        }
    }

    /// Creates a transform from a matrix.
    ///
    /// The matrix must be invertible. If the matrix is not invertible, a `NonInvertibleMatrixError` is returned.
    pub fn from_matrix(matrix: Matrix3x3<S>) -> Result<Transform2<S>, NonInvertibleMatrixError> {
        Ok(Transform2 {
            forward: matrix.clone(),
            inverse: matrix.inverted()?,
        })
    }

    /// Computes and returns a composite transform which first applies this and then the other transform.
    pub fn and_then(&self, next: &Transform2<S>) -> Transform2<S> {
        Transform2 {
            forward: &next.forward * &self.forward,
            inverse: &self.inverse * &next.inverse,
        }
    }

    /// Returns the inverse of this transform.
    pub fn inverted(&self) -> Transform2<S> {
        Transform2 {
            forward: self.inverse.clone(),
            inverse: self.forward.clone(),
        }
    }

    /// Inverts this transform.
    pub fn invert(&mut self) {
        *self = self.inverted();
    }
}

impl<S: Scalar> Transform<Vector2<S>> for Transform2<S> {
    type Output = Vector2<S>;

    /// Transforms a vector.
    fn transform(&self, vector: Vector2<S>) -> Vector2<S> {
        &self.forward * vector
    }

    /// Transforms a vector with the inverse transform.
    fn inverse_transform(&self, vector: Vector2<S>) -> Vector2<S> {
        &self.inverse * vector
    }
}

impl<S: Scalar> Transform<Point2<S>> for Transform2<S> {
    type Output = Point2<S>;

    /// Transforms a point.
    fn transform(&self, point: Point2<S>) -> Point2<S> {
        &self.forward * point
    }

    /// Transforms a point with the inverse transform.
    fn inverse_transform(&self, point: Point2<S>) -> Point2<S> {
        &self.inverse * point
    }
}

impl<S: Scalar> Transform<Ray2<S>> for Transform2<S> {
    type Output = Ray2<S>;

    /// Transforms a ray.
    fn transform(&self, ray: Ray2<S>) -> Ray2<S> {
        let origin = self.transform(ray.origin);
        let direction = self.transform(ray.direction);

        // Scale the range by the scale factor of the transform
        let scale = direction.length() / ray.direction.length();
        let range = (ray.range.start * scale)..(ray.range.end * scale);

        Ray2 { origin, direction, range }
    }

    /// Transforms a ray with the inverse transform.
    fn inverse_transform(&self, ray: Ray2<S>) -> Ray2<S> {
        let origin = self.inverse_transform(ray.origin);
        let direction = self.inverse_transform(ray.direction);

        // Scale the range by the scale factor of the transform
        let scale = direction.length() / ray.direction.length();
        let range = (ray.range.start * scale)..(ray.range.end * scale);

        Ray2 { origin, direction, range }
    }
}

impl<S: Scalar> Transform<BoundingBox2<S>> for Transform2<S> {
    type Output = BoundingBox2<S>;

    /// Transforms a bounding box.
    fn transform(&self, bb: BoundingBox2<S>) -> BoundingBox2<S> {
        BoundingBox2::from_corner_and_diagonal(self.transform(bb.min), self.transform(bb.diagonal()))
    }

    /// Transforms a bounding box with the inverse transform.
    fn inverse_transform(&self, bb: BoundingBox2<S>) -> BoundingBox2<S> {
        BoundingBox2::from_corner_and_diagonal(self.inverse_transform(bb.min), self.inverse_transform(bb.diagonal()))
    }
}

// ===== Transform3 ============================================================================================================================================

impl<S: Scalar> Transform3<S> {
    /// The identity transform.
    pub const IDENTITY: Transform3<S> = Transform3 {
        forward: Matrix4x4::IDENTITY,
        inverse: Matrix4x4::IDENTITY,
    };

    /// Returns the identity transform.
    pub fn identity() -> Transform3<S> {
        Transform3::IDENTITY
    }

    /// Returns `true` if this transform is equal to the identity transform, `false` otherwise.
    pub fn is_identity(&self) -> bool {
        *self == Transform3::IDENTITY
    }

    /// Returns a transform for translation in 3D space.
    pub fn translate(vector: Vector3<S>) -> Transform3<S> {
        Transform3 {
            forward: Matrix4x4::translate(vector),
            inverse: Matrix4x4::translate(-vector),
        }
    }

    /// Returns a transform for rotating around the X axis in 3D space.
    pub fn rotate_x(angle: Angle<S>) -> Transform3<S> {
        Transform3 {
            forward: Matrix4x4::rotate_x(angle),
            inverse: Matrix4x4::rotate_x(-angle),
        }
    }

    /// Returns a transform for rotating around the Y axis in 3D space.
    pub fn rotate_y(angle: Angle<S>) -> Transform3<S> {
        Transform3 {
            forward: Matrix4x4::rotate_y(angle),
            inverse: Matrix4x4::rotate_y(-angle),
        }
    }

    /// Returns a transform for rotating around the Z axis in 3D space.
    pub fn rotate_z(angle: Angle<S>) -> Transform3<S> {
        Transform3 {
            forward: Matrix4x4::rotate_z(angle),
            inverse: Matrix4x4::rotate_z(-angle),
        }
    }

    /// Returns a transform for rotating around an axis in 3D space.
    pub fn rotate_axis(axis: Vector3<S>, angle: Angle<S>) -> Transform3<S> {
        Transform3 {
            forward: Matrix4x4::rotate_axis(axis, angle),
            inverse: Matrix4x4::rotate_axis(axis, -angle),
        }
    }

    /// Returns a transform for scaling in 3D space in the X, Y and Z directions.
    pub fn scale(factor_x: S, factor_y: S, factor_z: S) -> Transform3<S> {
        Transform3 {
            forward: Matrix4x4::scale(factor_x, factor_y, factor_z),
            inverse: Matrix4x4::scale(factor_x.recip(), factor_y.recip(), factor_z.recip()),
        }
    }

    /// Returns a transform for scaling uniformly in 3D space.
    pub fn scale_uniform(factor: S) -> Transform3<S> {
        Transform3 {
            forward: Matrix4x4::scale_uniform(factor),
            inverse: Matrix4x4::scale_uniform(factor.recip()),
        }
    }

    /// Returns a "look at" transform. When applied to a camera, the camera will look from the point `from` to the point `target`,
    /// with `up` as the up direction.
    pub fn look_at(from: Point3<S>, target: Point3<S>, up: Vector3<S>) -> Result<Transform3<S>, NonInvertibleMatrixError> {
        let forward = Matrix4x4::look_at(from, target, up);
        let inverse = forward.inverted()?;

        Ok(Transform3 { forward, inverse })
    }

    /// Creates a transform from a matrix.
    ///
    /// The matrix must be invertible. If the matrix is not invertible, a `NonInvertibleMatrixError` is returned.
    pub fn from_matrix(matrix: Matrix4x4<S>) -> Result<Transform3<S>, NonInvertibleMatrixError> {
        Ok(Transform3 {
            forward: matrix.clone(),
            inverse: matrix.inverted()?,
        })
    }

    /// Computes and returns a composite transform which first applies this and then the other transform.
    pub fn and_then(&self, next: &Transform3<S>) -> Transform3<S> {
        Transform3 {
            forward: &next.forward * &self.forward,
            inverse: &self.inverse * &next.inverse,
        }
    }

    /// Returns the inverse of this transform.
    pub fn inverted(&self) -> Transform3<S> {
        Transform3 {
            forward: self.inverse.clone(),
            inverse: self.forward.clone(),
        }
    }

    /// Inverts this transform.
    pub fn invert(&mut self) {
        *self = self.inverted();
    }
}

impl<S: Scalar> Transform<Vector3<S>> for Transform3<S> {
    type Output = Vector3<S>;

    /// Transforms a vector.
    fn transform(&self, vector: Vector3<S>) -> Vector3<S> {
        &self.forward * vector
    }

    /// Transforms a vector with the inverse transform.
    fn inverse_transform(&self, vector: Vector3<S>) -> Vector3<S> {
        &self.inverse * vector
    }
}

impl<S: Scalar> Transform<Point3<S>> for Transform3<S> {
    type Output = Point3<S>;

    /// Transforms a point.
    fn transform(&self, point: Point3<S>) -> Point3<S> {
        &self.forward * point
    }

    /// Transforms a point with the inverse transform.
    fn inverse_transform(&self, point: Point3<S>) -> Point3<S> {
        &self.inverse * point
    }
}

impl<S: Scalar> Transform<Normal3<S>> for Transform3<S> {
    type Output = Normal3<S>;

    /// Transforms a normal.
    ///
    /// Transforming a normal works differently than transforming a vector.
    /// See [Physically Based Rendering: Applying Transformations - Normals](https://www.pbr-book.org/4ed/Geometry_and_Transformations/Applying_Transformations#Normals).
    fn transform(&self, normal: Normal3<S>) -> Normal3<S> {
        // NOTE: Normals need to be transformed with the transpose of the inverse.
        // See: https://www.pbr-book.org/4ed/Geometry_and_Transformations/Applying_Transformations#Normals
        &self.inverse.transposed() * normal
    }

    /// Transforms a normal with the inverse transform.
    ///
    /// Transforming a normal works differently than transforming a vector.
    /// See [Physically Based Rendering: Applying Transformations - Normals](https://www.pbr-book.org/4ed/Geometry_and_Transformations/Applying_Transformations#Normals).
    fn inverse_transform(&self, normal: Normal3<S>) -> Normal3<S> {
        // NOTE: Normals need to be transformed with the transpose of the inverse.
        // See: https://www.pbr-book.org/4ed/Geometry_and_Transformations/Applying_Transformations#Normals
        &self.forward.transposed() * normal
    }
}

impl<S: Scalar> Transform<Ray3<S>> for Transform3<S> {
    type Output = Ray3<S>;

    /// Transforms a ray.
    fn transform(&self, ray: Ray3<S>) -> Ray3<S> {
        let origin = self.transform(ray.origin);
        let direction = self.transform(ray.direction);

        // Scale the range by the scale factor of the transform
        let scale = direction.length() / ray.direction.length();
        let range = (ray.range.start * scale)..(ray.range.end * scale);

        Ray3 { origin, direction, range }
    }

    /// Transforms a ray with the inverse transform.
    fn inverse_transform(&self, ray: Ray3<S>) -> Ray3<S> {
        let origin = self.inverse_transform(ray.origin);
        let direction = self.inverse_transform(ray.direction);

        // Scale the range by the scale factor of the transform
        let scale = direction.length() / ray.direction.length();
        let range = (ray.range.start * scale)..(ray.range.end * scale);

        Ray3 { origin, direction, range }
    }
}

impl<S: Scalar> Transform<BoundingBox3<S>> for Transform3<S> {
    type Output = BoundingBox3<S>;

    /// Transforms a bounding box.
    fn transform(&self, bb: BoundingBox3<S>) -> BoundingBox3<S> {
        BoundingBox3::from_corner_and_diagonal(self.transform(bb.min), self.transform(bb.diagonal()))
    }

    /// Transforms a bounding box with the inverse transform.
    fn inverse_transform(&self, bb: BoundingBox3<S>) -> BoundingBox3<S> {
        BoundingBox3::from_corner_and_diagonal(self.inverse_transform(bb.min), self.inverse_transform(bb.diagonal()))
    }
}
