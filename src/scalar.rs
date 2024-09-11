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

use crate::{Distance, Length, MinMax, RelativeDistance, RelativeLength};
use num_traits::float::TotalOrder;
use num_traits::{ConstOne, ConstZero, Float, FloatConst, NumAssignOps};
use std::cmp::Ordering;

/// A scalar value.
pub trait Scalar: Float + ConstZero + ConstOne + FloatConst + NumAssignOps + TotalOrder {}

// ===== Scalar ================================================================================================================================================

impl Scalar for f32 {}

impl Scalar for f64 {}

impl<S: Scalar> MinMax for S {
    fn min(self, other: Self) -> Self {
        S::min(self, other)
    }

    fn max(self, other: Self) -> Self {
        S::max(self, other)
    }

    fn min_max(self, other: Self) -> (Self, Self) {
        if self.total_cmp(&other) == Ordering::Less { (self, other) } else { (other, self) }
    }
}

impl<S: Scalar> Length<S> for S {
    fn length(self) -> S {
        self.abs()
    }
}

impl<S: Scalar> RelativeLength<S> for S {
    fn cmp_length(self, other: Self) -> Ordering {
        self.length().total_cmp(&other.length())
    }
}

impl<S: Scalar> Distance<S> for S {
    fn distance(self, other: Self) -> S {
        (self - other).length()
    }
}

impl<S: Scalar> RelativeDistance<S> for S {
    fn cmp_distance(self, first: Self, second: Self) -> Ordering {
        (self - first).cmp_length(self - second)
    }
}
