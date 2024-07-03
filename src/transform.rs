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

/// Trait to be implemented for types that can transform a value of type `T`.
pub trait Transform<T> {
    /// The output type that results from transforming a value of type `T`.
    type Output;

    /// Transforms a value of type `T`.
    fn transform(&self, value: T) -> Self::Output;
}

/// Error returned when computing the inverse of a singular matrix is attempted.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub struct NonInvertibleMatrixError;
