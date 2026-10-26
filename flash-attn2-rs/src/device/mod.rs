//! Device side: every `#[kernel]` the host launches by name, plus the shared tile primitives.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

mod backward;
mod cache;
mod config;
mod dropout;
mod forward;
mod inlined;
mod layout;
mod rng;
mod varlen;
