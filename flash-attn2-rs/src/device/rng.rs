use cuda_device::{DisjointSlice, kernel, launch_bounds};

// Resolve PyTorch's graph-safe Philox descriptor on the launch stream.
#[kernel]
#[launch_bounds(1)]
pub fn resolve_rng_state(
    state: DisjointSlice<u64>,
    seed: u64,
    offset: u64,
    intragraph: u64,
    captured: u32,
) {
    let mut state = state;
    unsafe {
        let (seed, offset) = if captured != 0 {
            (
                *(seed as *const u64),
                (*(offset as *const u64)).wrapping_add(intragraph),
            )
        } else {
            (seed, offset)
        };
        *state.as_mut_ptr() = seed;
        *state.as_mut_ptr().add(1) = offset;
    }
}
