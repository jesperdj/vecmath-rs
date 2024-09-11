# Vector math for 2D and 3D graphics applications.

The main structs in this crate are `Vector2`, `Vector3`, `Point2`, `Point3`, `Transform2` and `Transform3`.
These are used to represent vectors, points and transformations in 2D and 3D space.
They are generic over the element type, which must conform to the trait `Scalar`. In practice this means you can create vectors, points and transforms
with `f32` and `f64` elements.

Other structs are `Normal3` to represent surface normals in 3D space, `Angle` to represent angles in radians or degrees, `Ray2` and `Ray3` which represent
rays and `BoundingBox2` and `BoundingBox3` which represent axis-aligned bounding boxes.

There are also `Matrix3x3` and `Matrix4x4` structs for transformation matrices. For transformations, you should however use `Transform2` and `Transform3`
which provide convenient `transform()` methods for transforming vectors, points, normals, rays and bounding boxes.
