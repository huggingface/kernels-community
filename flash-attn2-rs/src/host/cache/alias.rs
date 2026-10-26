//! Bounded proofs that two cache views share no bytes.

use tvm_ffi::Tensor;

use crate::host::layout::{
    layout_rectangle_count, nonoverlapping_bounded_differences, nonoverlapping_small_layout,
    overlaps, tensor_span,
};

// Prove disjoint packed views without enumerating or reading GPU data.
pub(super) fn disjoint_views(a: &Tensor, b: &Tensor) -> bool {
    disjoint_views_proven(a, b)
        || disjoint_finite_rows(a, b)
        || disjoint_small_views(a, b)
        || disjoint_periodic_views(a, b)
        || disjoint_drifting_axes(a, b)
        || disjoint_grouped_rows(a, b)
}

// Collapse dense inner dims and regular outer rows. Units are bytes.
fn cache_regular_rows(t: &Tensor) -> Option<(i128, i128, i128, i128)> {
    cache_regular_rows_below(t, None)
}

fn cache_regular_rows_below(
    t: &Tensor,
    threshold: Option<u64>,
) -> Option<(i128, i128, i128, i128)> {
    let bytes = u64::from(t.dtype().bits) / 8;
    if bytes == 0 {
        return None;
    }
    let mut dims: Vec<_> = t
        .shape()
        .iter()
        .zip(t.strides())
        .filter(|&(&n, &s)| n > 1 && s != 0)
        .map(|(&n, &s)| Some((u64::try_from(s).ok()?.checked_mul(bytes)?, n as u64)))
        .collect::<Option<_>>()?;
    dims.retain(|&(stride, _)| threshold.is_none_or(|limit| stride < limit));
    dims.sort_unstable();
    let (mut width, mut i) = (bytes, 0);
    while i < dims.len() && dims[i].0 == width {
        width = width.checked_mul(dims[i].1)?;
        i += 1;
    }
    let (step, mut count) = dims.get(i).copied().unwrap_or((1, 1));
    for &(stride, n) in dims.iter().skip(i + 1) {
        if stride != step.checked_mul(count)? {
            return None;
        }
        count = count.checked_mul(n)?;
    }
    // Bounded so the i128 rectangle-count products cannot overflow.
    if count > i64::MAX as u64 {
        return None;
    }
    let start = t.data_ptr() as u64;
    start
        .checked_add(step.checked_mul(count - 1)?)?
        .checked_add(width)?;
    Some((start as i128, step as i128, count as i128, width as i128))
}

// Reverse B's row index to count overlaps as a positive-stride rectangle.
fn disjoint_finite_rows(a: &Tensor, b: &Tensor) -> bool {
    if a.numel() == 0 || b.numel() == 0 {
        return true;
    }
    let (Some((ap, astep, an, aw)), Some((bp, bstep, bn, bw))) =
        (cache_regular_rows(a), cache_regular_rows(b))
    else {
        return false;
    };
    let base = ap - bp - bstep * (bn - 1);
    layout_rectangle_count(astep, an, bstep, bn, bw - 1 - base)
        == layout_rectangle_count(astep, an, bstep, bn, -aw - base)
}

// Bound relative drift across padded outer axes.
fn disjoint_drifting_axes(a: &Tensor, b: &Tensor) -> bool {
    if a.shape().len() != 4 || a.shape() != b.shape() || a.shape().iter().any(|&n| n <= 0) {
        return false;
    }
    let strides = |t: &Tensor| -> Option<[u64; 4]> {
        let bytes = u64::from(t.dtype().bits) / 8;
        if bytes == 0 {
            return None;
        }
        let mut steps = [0; 4];
        let mut span = bytes;
        for i in 0..4 {
            steps[i] = u64::try_from(t.strides()[i]).ok()?.checked_mul(bytes)?;
            span = span.checked_add(steps[i].checked_mul(t.shape()[i] as u64 - 1)?)?;
        }
        (t.data_ptr() as u64).checked_add(span)?;
        Some(steps)
    };
    let (Some(ast), Some(bst)) = (strides(a), strides(b)) else {
        return false;
    };
    // At most four stride thresholds, independent of every dimension size.
    for axis in 0..4 {
        if a.shape()[axis] <= 1 {
            continue;
        }
        let threshold = ast[axis].min(bst[axis]);
        if threshold == 0 {
            continue;
        }
        let mut outer = [(0u64, 0u64); 4];
        let (mut count, mut low, mut high) = (0, 0i128, 0i128);
        let mut compatible = true;
        for i in 0..4 {
            let n = a.shape()[i] as u64;
            if n <= 1 {
                continue;
            }
            let (ao, bo) = (ast[i] >= threshold, bst[i] >= threshold);
            if ao != bo {
                compatible = false;
                break;
            }
            if ao {
                outer[count] = (ast[i], n);
                count += 1;
                let drift = (bst[i] as i128 - ast[i] as i128) * (n - 1) as i128;
                low += drift.min(0);
                high += drift.max(0);
            }
        }
        if !compatible || count == 0 {
            continue;
        }
        let (Some((ap, astep, an, aw)), Some((bp, bstep, bn, bw))) = (
            cache_regular_rows_below(a, Some(threshold)),
            cache_regular_rows_below(b, Some(threshold)),
        ) else {
            continue;
        };
        low += bp - ap;
        high += bp - ap;
        let (aspan, bspan) = (astep * (an - 1) + aw, bstep * (bn - 1) + bw);
        let mut envelope = aspan.max(high + bspan) - low.min(0);
        outer[..count].sort_unstable();
        for &(step, n) in &outer[..count] {
            if envelope > step as i128 {
                compatible = false;
                break;
            }
            envelope += step as i128 * (n - 1) as i128;
        }
        if !compatible {
            continue;
        }
        let width = bw + high - low;
        let base = -low - bstep * (bn - 1);
        if layout_rectangle_count(astep, an, bstep, bn, width - 1 - base)
            == layout_rectangle_count(astep, an, bstep, bn, -aw - base)
        {
            return true;
        }
    }
    false
}

// Fallback: keep the largest padded axis symbolic, enumerate <= 8 groups.
fn cache_grouped_rows(t: &Tensor) -> Option<(Vec<u64>, i128, i128, i128)> {
    let bytes = u64::from(t.dtype().bits) / 8;
    if bytes == 0 {
        return None;
    }
    let mut dims: Vec<_> = t
        .shape()
        .iter()
        .zip(t.strides())
        .filter(|&(&n, &s)| n > 1 && s != 0)
        .map(|(&n, &s)| Some((u64::try_from(s).ok()?.checked_mul(bytes)?, n as u64)))
        .collect::<Option<_>>()?;
    dims.sort_unstable();
    let (mut width, mut i) = (bytes, 0);
    while i < dims.len() && dims[i].0 == width {
        width = width.checked_mul(dims[i].1)?;
        i += 1;
    }
    let mut axes = dims[i..].to_vec();
    let (step, count) = match axes.iter().enumerate().max_by_key(|(_, axis)| axis.1) {
        Some((index, _)) => axes.remove(index),
        None => (1, 1),
    };
    if count > i64::MAX as u64 {
        return None;
    }
    let mut groups = 1u64;
    for &(_, n) in &axes {
        groups = groups.checked_mul(n)?;
        if groups > 8 {
            return None;
        }
    }
    let mut starts = vec![t.data_ptr() as u64];
    for (stride, n) in axes {
        let mut next = Vec::with_capacity(starts.len() * n as usize);
        for start in starts {
            for j in 0..n {
                next.push(start.checked_add(j.checked_mul(stride)?)?);
            }
        }
        starts = next;
    }
    let span = step.checked_mul(count - 1)?.checked_add(width)?;
    for &start in &starts {
        start.checked_add(span)?;
    }
    Some((starts, step as i128, count as i128, width as i128))
}

fn disjoint_grouped_rows(a: &Tensor, b: &Tensor) -> bool {
    if a.numel() == 0 || b.numel() == 0 {
        return true;
    }
    let (Some((a, astep, an, aw)), Some((b, bstep, bn, bw))) =
        (cache_grouped_rows(a), cache_grouped_rows(b))
    else {
        return false;
    };
    for ap in a {
        for &bp in &b {
            let (ap, bp) = (ap as i128, bp as i128);
            if ap + astep * (an - 1) + aw <= bp || bp + bstep * (bn - 1) + bw <= ap {
                continue;
            }
            let base = ap - bp - bstep * (bn - 1);
            if layout_rectangle_count(astep, an, bstep, bn, bw - 1 - base)
                != layout_rectangle_count(astep, an, bstep, bn, -aw - base)
            {
                return false;
            }
        }
    }
    true
}

// Exact rare-path check for small address sets; reads no tensor data.
fn cache_view_rows(t: &Tensor, threshold: Option<u64>) -> Option<(Vec<u64>, u64)> {
    let element_bytes = u64::from(t.dtype().bits) / 8;
    let mut dims: Vec<_> = t
        .shape()
        .iter()
        .zip(t.strides())
        .filter(|&(&n, &s)| n > 1 && s != 0)
        .map(|(&n, &s)| Some((u64::try_from(s).ok()?.checked_mul(element_bytes)?, n as u64)))
        .collect::<Option<_>>()?;
    dims.retain(|&(stride, _)| threshold.is_none_or(|limit| stride < limit));
    dims.sort_unstable();
    let (mut width, mut i) = (element_bytes, 0usize);
    while i < dims.len() && dims[i].0 == width {
        width = width.checked_mul(dims[i].1)?;
        i += 1;
    }
    let mut count = 1u64;
    for &(_, n) in &dims[i..] {
        count = count.checked_mul(n)?;
        if count > 4096 {
            return None;
        }
    }
    let mut starts = vec![t.data_ptr() as u64];
    for &(stride, n) in &dims[i..] {
        let mut next = Vec::with_capacity(starts.len() * n as usize);
        for start in starts {
            for j in 0..n {
                next.push(start.checked_add(j.checked_mul(stride)?)?);
            }
        }
        starts = next;
    }
    starts.sort_unstable();
    Some((starts, width))
}

fn disjoint_small_views(a: &Tensor, b: &Tensor) -> bool {
    let (Some((a, aw)), Some((b, bw))) = (cache_view_rows(a, None), cache_view_rows(b, None))
    else {
        return false;
    };
    let (mut i, mut j) = (0usize, 0usize);
    while i < a.len() && j < b.len() {
        let (Some(a_end), Some(b_end)) = (a[i].checked_add(aw), b[j].checked_add(bw)) else {
            return false;
        };
        if a_end <= b[j] {
            i += 1;
        } else if b_end <= a[i] {
            j += 1;
        } else {
            return false;
        }
    }
    true
}

// Prove periodic separation without enumerating the inner rows.
fn disjoint_regular_residues(a: &Tensor, b: &Tensor, threshold: u64, period: u64) -> bool {
    let (Some((ap, astep, an, aw)), Some((bp, bstep, bn, bw))) = (
        cache_regular_rows_below(a, Some(threshold)),
        cache_regular_rows_below(b, Some(threshold)),
    ) else {
        return false;
    };
    let period = period as i128;
    if astep * (an - 1) + aw > period || bstep * (bn - 1) + bw > period {
        return false;
    }
    let shift = (bp - ap).rem_euclid(period);
    for shift in [shift, shift - period] {
        let base = -shift - bstep * (bn - 1);
        if layout_rectangle_count(astep, an, bstep, bn, bw - 1 - base)
            != layout_rectangle_count(astep, an, bstep, bn, -aw - base)
        {
            return false;
        }
    }
    true
}

// Disjoint residues of a shared period make every repetition safe.
fn disjoint_periodic_views(a: &Tensor, b: &Tensor) -> bool {
    // Byte units so int32 metadata and half caches share the proof.
    let mut strides = Vec::new();
    for t in [a, b] {
        let bytes = u64::from(t.dtype().bits) / 8;
        for (&n, &s) in t.shape().iter().zip(t.strides()) {
            if n > 1 && s != 0 {
                let Some(stride) = u64::try_from(s).ok().and_then(|s| s.checked_mul(bytes)) else {
                    return false;
                };
                strides.push(stride);
            }
        }
    }
    let mut thresholds = strides.clone();
    thresholds.sort_unstable_by(|a, b| b.cmp(a));
    thresholds.dedup();
    for threshold in thresholds {
        let mut period = 0u64;
        for &stride in &strides {
            if stride >= threshold {
                let mut value = stride;
                while value != 0 {
                    (period, value) = (value, period % value);
                }
            }
        }
        if period == 0 {
            continue;
        }
        if disjoint_regular_residues(a, b, threshold, period) {
            return true;
        }
        let residues = |t| -> Option<Vec<(u64, u64)>> {
            let (starts, width) = cache_view_rows(t, Some(threshold))?;
            if width >= period {
                return None;
            }
            let mut intervals = Vec::with_capacity(starts.len() * 2);
            for start in starts {
                let start = start % period;
                let end = start.checked_add(width)?;
                intervals.push((start, end.min(period)));
                if end > period {
                    intervals.push((0, end - period));
                }
            }
            intervals.sort_unstable();
            Some(intervals)
        };
        let (Some(left), Some(right)) = (residues(a), residues(b)) else {
            continue;
        };
        let (mut i, mut j) = (0usize, 0usize);
        while i < left.len() && j < right.len() {
            if left[i].1 <= right[j].0 {
                i += 1;
            } else if right[j].1 <= left[i].0 {
                j += 1;
            } else {
                break;
            }
        }
        if i == left.len() || j == right.len() {
            return true;
        }
    }
    false
}

// Narrow an auxiliary interval through cache-axis bounds, conservatively.
fn disjoint_auxiliary_interval(t: &Tensor, cache: &Tensor) -> bool {
    let Some((start, step, rows, row_width)) = cache_regular_rows(t) else {
        return false;
    };
    if cache.shape().len() != 4 || cache.shape().iter().any(|&n| n <= 0) {
        return false;
    }
    let bytes = u64::from(cache.dtype().bits) / 8;
    if bytes == 0 {
        return false;
    }
    let mut axes = [(0u64, 0u64); 4];
    let mut count = 0;
    for (&n, &stride) in cache.shape().iter().zip(cache.strides()) {
        let Some(step) = u64::try_from(stride)
            .ok()
            .and_then(|s| s.checked_mul(bytes))
        else {
            return false;
        };
        if n > 1 && step != 0 {
            axes[count] = (step, n as u64);
            count += 1;
        }
    }
    axes[..count].sort_unstable();
    let mut spans = [bytes; 5];
    for i in 0..count {
        let (step, n) = axes[i];
        let Some(span) = step
            .checked_mul(n - 1)
            .and_then(|s| s.checked_add(spans[i]))
        else {
            return false;
        };
        spans[i + 1] = span;
    }
    let origin = cache.data_ptr() as u64;
    if origin.checked_add(spans[count]).is_none() {
        return false;
    }
    // The regular-row descriptor already checks this span.
    let end = start + step * (rows - 1) + row_width;
    let separated = |start: i128, end: i128, axes: &[(u64, u64)], spans: &[u64]| {
        let mut base = origin as i128;
        for i in (0..axes.len()).rev() {
            let (step, n) = axes[i];
            let step = step as i128;
            // A group's half-open bounding interval must intersect [start,end).
            let first = (start - base - spans[i] as i128).div_euclid(step) + 1;
            let last = (end - 1 - base).div_euclid(step);
            let (first, last) = (first.max(0), last.min(n as i128 - 1));
            if first > last {
                return true;
            }
            if first != last {
                return false;
            }
            base += first * step;
        }
        end <= base || base + bytes as i128 <= start
    };
    if separated(start, end, &axes[..count], &spans) {
        return true;
    }
    // Pair one auxiliary row with each cache batch, widened by drift.
    if rows <= 1 || rows != cache.shape()[0] as i128 {
        return false;
    }
    let Some(batch_step) = u64::try_from(cache.strides()[0])
        .ok()
        .and_then(|s| s.checked_mul(bytes))
    else {
        return false;
    };
    let Some(batch_axis) = axes[..count]
        .iter()
        .position(|&(s, n)| s == batch_step && n as i128 == rows)
    else {
        return false;
    };
    for i in batch_axis..count - 1 {
        axes[i] = axes[i + 1];
    }
    count -= 1;
    // These subset sums/products are bounded by the checked full cache span.
    for i in 0..count {
        spans[i + 1] = spans[i] + axes[i].0 * (axes[i].1 - 1);
    }
    let drift = (step - batch_step as i128) * (rows - 1);
    let low = start + drift.min(0);
    let high = start + drift.max(0) + row_width;
    let base = origin as i128;
    if (base + spans[count] as i128).max(high) - base.min(low) > batch_step as i128 {
        return false;
    }
    separated(low, high, &axes[..count], &spans)
}

// Auxiliary tensors (int32 metadata, FP16/BF16 rotary) compared by bytes.
pub(super) fn disjoint_auxiliary(t: &Tensor, cache: &Tensor) -> bool {
    if t.numel() == 0 || cache.numel() == 0 {
        return true;
    }
    let (start, cache_start) = (t.data_ptr() as u64, cache.data_ptr() as u64);
    let end = tensor_span(t)
        .checked_mul(u64::from(t.dtype().bits) / 8)
        .and_then(|size| start.checked_add(size));
    let cache_end = tensor_span(cache)
        .checked_mul(2)
        .and_then(|size| cache_start.checked_add(size));
    let (Some(end), Some(cache_end)) = (end, cache_end) else {
        return false;
    };
    end <= cache_start
        || cache_end <= start
        || disjoint_finite_rows(t, cache)
        || disjoint_small_views(t, cache)
        || disjoint_periodic_views(t, cache)
        || disjoint_drifting_axes(t, cache)
        || disjoint_grouped_rows(t, cache)
        || disjoint_auxiliary_interval(t, cache)
}

fn disjoint_views_proven(a: &Tensor, b: &Tensor) -> bool {
    if a.numel() == 0 || b.numel() == 0 || !overlaps(a, b) {
        return true;
    }
    let (lo, hi) = if (a.data_ptr() as u64) < b.data_ptr() as u64 {
        (a, b)
    } else {
        (b, a)
    };
    let distance = (hi.data_ptr() as u64 - lo.data_ptr() as u64) / 2;
    for split in 1..4 {
        // Disjoint residues of the common period: no aliasing.
        let mut period = 0u64;
        for t in [lo, hi] {
            for (&n, &stride) in t.shape()[..split].iter().zip(&t.strides()[..split]) {
                if n > 1 {
                    let mut value = stride as u64;
                    while value != 0 {
                        (period, value) = (value, period % value);
                    }
                }
            }
        }
        if period == 0 {
            continue;
        }
        let inner_span = |t: &Tensor| -> Option<u64> {
            t.shape()[split..]
                .iter()
                .zip(&t.strides()[split..])
                .try_fold(1u64, |span, (&n, &stride)| {
                    (n as u64 - 1).checked_mul(stride as u64)?.checked_add(span)
                })
        };
        let (Some(lo_span), Some(hi_span)) = (inner_span(lo), inner_span(hi)) else {
            continue;
        };
        let separation = distance % period;
        if lo_span <= separation && hi_span <= period - separation {
            return true;
        }
    }
    // Split each view at a shared stride threshold; disjoint residues separate.
    for &threshold in lo.strides().iter().chain(hi.strides()) {
        if threshold <= 1 {
            continue;
        }
        let mut period = 0u64;
        let mut spans = [Some(1u64); 2];
        for (i, t) in [lo, hi].iter().enumerate() {
            for (&n, &stride) in t.shape().iter().zip(t.strides()) {
                if n <= 1 {
                    continue;
                }
                if stride >= threshold {
                    let mut value = stride as u64;
                    while value != 0 {
                        (period, value) = (value, period % value);
                    }
                } else {
                    spans[i] = spans[i].and_then(|span| {
                        (n as u64 - 1).checked_mul(stride as u64)?.checked_add(span)
                    });
                }
            }
        }
        let [Some(lo_span), Some(hi_span)] = spans else {
            continue;
        };
        if period != 0 {
            let separation = distance % period;
            if lo_span <= separation && hi_span <= period - separation {
                return true;
            }
        }
    }
    // Canonicalize axis order and singletons before the union proof.
    let physical_axes = |t: &Tensor| {
        let mut dims: Vec<(u64, u64)> = t
            .shape()
            .iter()
            .zip(t.strides())
            .filter(|&(&n, _)| n > 1)
            .map(|(&n, &stride)| (stride as u64, n as u64))
            .collect();
        dims.sort_unstable();
        dims
    };
    let axes = physical_axes(a);
    if axes != physical_axes(b) {
        if a.shape() != b.shape() {
            return false;
        }
        // Disjoint inner regions under shared outer strides never overlap.
        let (lo, hi) = if (a.data_ptr() as u64) < b.data_ptr() as u64 {
            (a, b)
        } else {
            (b, a)
        };
        let distance = (hi.data_ptr() as u64 - lo.data_ptr() as u64) / 2;
        for split in 1..4 {
            if lo.strides()[..split] != hi.strides()[..split] {
                continue;
            }
            let inner_span = |t: &Tensor| -> Option<u64> {
                t.shape()[split..]
                    .iter()
                    .zip(&t.strides()[split..])
                    .try_fold(1u64, |span, (&n, &stride)| {
                        (n as u64 - 1).checked_mul(stride as u64)?.checked_add(span)
                    })
            };
            let (Some(lo_span), Some(hi_span)) = (inner_span(lo), inner_span(hi)) else {
                continue;
            };
            if lo_span > distance {
                continue;
            }
            let Some(mut span) = distance.checked_add(hi_span) else {
                continue;
            };
            let mut dims: Vec<_> = lo.shape()[..split]
                .iter()
                .zip(&lo.strides()[..split])
                .filter(|&(&n, _)| n > 1)
                .map(|(&n, &stride)| (stride as u64, n as u64))
                .collect();
            dims.sort_unstable();
            let mut disjoint = true;
            for (stride, n) in dims {
                if stride < span {
                    disjoint = false;
                    break;
                }
                let Some(next) = (n - 1)
                    .checked_mul(stride)
                    .and_then(|v| v.checked_add(span))
                else {
                    disjoint = false;
                    break;
                };
                span = next;
            }
            if disjoint {
                return true;
            }
        }
        return false;
    }
    let distance = (a.data_ptr() as u64).abs_diff(b.data_ptr() as u64);
    if distance % 2 != 0 {
        return false;
    }
    let mut dims = axes;
    dims.push((distance / 2, 2));
    dims.sort_unstable();
    let mut span = 1u64;
    for &(stride, n) in &dims {
        if stride < span {
            // Unique addresses in the union prove the views disjoint.
            let dims: Option<Vec<_>> = dims
                .iter()
                .map(|&(s, n)| Some((i64::try_from(s).ok()?, i64::try_from(n).ok()?)))
                .collect();
            return dims.is_some_and(|dims| {
                nonoverlapping_small_layout(&dims) || nonoverlapping_bounded_differences(&dims)
            });
        }
        let Some(next) = (n - 1)
            .checked_mul(stride)
            .and_then(|v| v.checked_add(span))
        else {
            return false;
        };
        span = next;
    }
    true
}
