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

/// Trait for types that have `min()` and `max()` methods.
pub trait MinMax: Copy {
    /// Compares and returns the minimum of two values.
    fn min(self, other: Self) -> Self;

    /// Compares and returns the maximum of two values.
    fn max(self, other: Self) -> Self;
}

/// Compares and returns the minimum of two values.
#[inline]
pub fn min<T: MinMax>(value: T, other: T) -> T {
    value.min(other)
}

/// Compares and returns the maximum of two values.
#[inline]
pub fn max<T: MinMax>(value: T, other: T) -> T {
    value.max(other)
}
