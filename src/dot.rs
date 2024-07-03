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

/// Trait for types for which a dot product between values can be computed.
pub trait DotProduct<U: Copy>: Copy {
    /// The output type which expresses the dot product between values.
    type Output;

    /// Computes and returns the dot product between two values.
    fn dot(self, other: U) -> Self::Output;
}

/// Computes and returns the dot product between two values.
#[inline]
pub fn dot<T: DotProduct<U>, U: Copy>(value: T, other: U) -> T::Output {
    value.dot(other)
}
