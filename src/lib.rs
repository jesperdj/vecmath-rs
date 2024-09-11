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

//! Vector math for 2D and 3D graphics applications.
//!
//! The main structs in this crate are [Vector2], [Vector3], [Point2], [Point3], [Transform2] and [Transform3].
//! These are used to represent vectors, points and transformations in 2D and 3D space.
//! They are generic over the element type, which must conform to the trait [Scalar]. In practice this means you can create vectors, points and transforms
//! with `f32` and `f64` elements.
//!
//! Other structs are [Normal3] to represent surface normals in 3D space, [Angle] to represent angles in radians or degrees, [Ray2] and [Ray3] which represent
//! rays and [BoundingBox2] and [BoundingBox3] which represent axis-aligned bounding boxes.
//!
//! There are also [Matrix3x3] and [Matrix4x4] structs for transformation matrices. For transformations, you should however use [Transform2] and [Transform3]
//! which provide convenient `transform()` methods for transforming vectors, points, normals, rays and bounding boxes.

mod angle;
mod boundingbox;
mod dimension;
mod distance;
mod dot;
mod length;
mod matrix;
mod minmax;
mod normal;
mod point;
mod ray;
mod scalar;
mod transform;
mod union;
mod vector;

pub use angle::*;
pub use boundingbox::*;
pub use dimension::*;
pub use distance::*;
pub use dot::*;
pub use length::*;
pub use matrix::*;
pub use minmax::*;
pub use normal::*;
pub use point::*;
pub use ray::*;
pub use scalar::*;
pub use transform::*;
pub use union::*;
pub use vector::*;
