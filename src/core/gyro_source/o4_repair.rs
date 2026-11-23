// SPDX-License-Identifier: GPL-3.0-or-later

use super::{Quat64, TimeQuat};
use serde::{Deserialize, Serialize};
use rustfft::{num_complex::Complex, FftPlanner};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RepairMode {
    #[default]
    Off,
    #[serde(alias = "doctor")]
    ShortSmoothing,
    #[serde(alias = "selective")]
    MediumSmoothing,
    /// Attenuate angular-rate departures from a local median, then reintegrate.
    RateAttenuation,
    /// Bridge only short, high-rate bursts between clean orientation anchors.
    BurstBridge,
    /// Suppress only large local orientation residuals while retaining slower turns.
    SelectiveResidual,
    Flatten,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RepairRegion {
    pub start_ms: f64,
    pub end_ms: f64,
    pub mode: RepairMode,
    /// 0..1, with 1 being the full effect of the selected mode.
    pub strength: f64,
}

impl RepairRegion {
    pub fn valid(&self) -> bool {
        self.start_ms.is_finite() && self.end_ms.is_finite()
            && self.start_ms >= 0.0 && self.end_ms > self.start_ms
            && self.strength.is_finite() && (0.0..=1.0).contains(&self.strength)
    }
}

/// Find sustained orientation innovations. This is only a motion-data candidate detector;
/// image disagreement still needs visual confirmation in the preview.
pub fn detect(quats: &TimeQuat, duration_ms: f64) -> Vec<RepairRegion> {
    if quats.len() < 8 || duration_ms <= 0.0 { return Vec::new(); }
    let samples: Vec<_> = quats.iter().map(|(t, q)| (*t, *q)).collect();
    // Frame-scale angular motion preserves the sustained pattern that makes stabilization wobble,
    // rather than reacting to every 1 kHz sample's ordinary sensor noise.
    let step_us = ((samples.last().unwrap().0 - samples.first().unwrap().0) as f64 / (samples.len() - 1) as f64).max(1.0);
    let frame_samples = (16_700.0 / step_us).round().max(1.0) as usize;
    let mut frame_motion = Vec::new();
    for i in (0..samples.len().saturating_sub(frame_samples)).step_by(frame_samples) {
        let dt = (samples[i + frame_samples].0 - samples[i].0) as f64 / 1_000_000.0;
        if dt <= 0.0 { continue; }
        let rate = (samples[i].1.inverse() * samples[i + frame_samples].1).scaled_axis() * (180.0 / std::f64::consts::PI / dt);
        frame_motion.push((samples[i].0, rate));
    }
    if frame_motion.len() < 8 { return Vec::new(); }
    let mut residuals = Vec::with_capacity(frame_motion.len());
    let radius = 5usize;
    for i in 0..frame_motion.len() {
        let from = i.saturating_sub(radius);
        let to = (i + radius + 1).min(frame_motion.len());
        let median = nalgebra::Vector3::from_fn(|axis, _| {
            let mut local = frame_motion[from..to].iter().map(|x| x.1[axis]).collect::<Vec<_>>();
            local.sort_by(f64::total_cmp);
            local[local.len() / 2]
        });
        residuals.push((frame_motion[i].1 - median).norm());
    }
    let mut sorted = residuals.clone();
    sorted.sort_by(f64::total_cmp);
    let median = sorted[sorted.len() / 2];
    let p95 = sorted[(sorted.len() * 95 / 100).min(sorted.len() - 1)];
    let threshold = (median * 8.0).max(p95 * 1.2).max(2.5);
    let mut spans: Vec<(f64, f64)> = Vec::new();
    for (i, &value) in residuals.iter().enumerate() {
        if value < threshold { continue; }
        let time_ms = frame_motion[i].0 as f64 / 1000.0;
        let start = (time_ms - 120.0).max(0.0);
        let end = (time_ms + 120.0).min(duration_ms);
        if let Some(last) = spans.last_mut() {
            if start <= last.1 + 80.0 { last.1 = last.1.max(end); continue; }
        }
        spans.push((start, end));
    }
    spans.into_iter().map(|(start_ms, end_ms)| RepairRegion {
        start_ms, end_ms, mode: RepairMode::Off, strength: 0.65,
    }).collect()
}

/// Apply one or more bounded, blended corrections to the integrated orientations.
/// The unedited quaternion track remains available to the comparison preview.
pub fn apply(quats: &mut TimeQuat, regions: &[RepairRegion]) {
    if quats.len() < 3 { return; }
    let original = quats.clone();
    for region in regions.iter().filter(|r| r.valid() && r.mode != RepairMode::Off && r.strength > 0.0) {
        let start = (region.start_ms * 1000.0).round() as i64;
        let end = (region.end_ms * 1000.0).round() as i64;
        let anchors: Vec<(i64, Quat64)> = original.range(..=start).next_back().into_iter()
            .chain(original.range(end..).next()).map(|(t, q)| (*t, *q)).collect();
        if anchors.len() != 2 || anchors[0].0 >= anchors[1].0 { continue; }
        if matches!(region.mode, RepairMode::RateAttenuation | RepairMode::BurstBridge | RepairMode::SelectiveResidual) {
            apply_adaptive(quats, &original, region, start, end);
            continue;
        }
        let window_us = match region.mode {
            RepairMode::ShortSmoothing => 80_000i64,
            RepairMode::MediumSmoothing => 250_000,
            RepairMode::Flatten => 1_500_000,
            _ => continue,
        };
        for (ts, output) in quats.range_mut(start..=end) {
            let source = original.get(ts).copied().unwrap_or(*output);
            let target = match region.mode {
                RepairMode::ShortSmoothing | RepairMode::MediumSmoothing => {
                    let left = original.range((ts - window_us)..=*ts).next().map(|(t, q)| (*t, *q));
                    let right = original.range(*ts..=(ts + window_us)).next_back().map(|(t, q)| (*t, *q));
                    match (left, right) {
                        (Some((a, qa)), Some((b, qb))) if b > a => qa.slerp(&qb, (*ts - a) as f64 / (b - a) as f64),
                        _ => source,
                    }
                }
                RepairMode::Flatten => anchors[0].1.slerp(&anchors[1].1, (*ts - anchors[0].0) as f64 / (anchors[1].0 - anchors[0].0) as f64),
                _ => source,
            };
            let edge = ((*ts - start).min(end - *ts) as f64 / 150_000.0).clamp(0.0, 1.0);
            let blend = region.strength * edge * edge * (3.0 - 2.0 * edge);
            *output = source.slerp(&target, blend);
        }
    }
}

fn median(values: &mut [f64]) -> f64 {
    if values.is_empty() { return 0.0; }
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn smooth_edge(t: i64, start: i64, end: i64) -> f64 {
    let x = ((t - start).min(end - t) as f64 / 100_000.0).clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

fn apply_adaptive(output: &mut TimeQuat, original: &TimeQuat, region: &RepairRegion, start: i64, end: i64) {
    use nalgebra::Vector3;
    let samples: Vec<_> = original.range(..=end).filter(|(t, _)| **t >= start.saturating_sub(150_000))
        .map(|(t, q)| (*t, *q)).collect();
    if samples.len() < 5 { return; }
    let first = samples.iter().position(|x| x.0 >= start).unwrap_or(0);
    let last = samples.iter().rposition(|x| x.0 <= end).unwrap_or(samples.len() - 1);
    if last <= first + 2 { return; }
    let mut rate = vec![Vector3::zeros(); samples.len()];
    for i in 1..samples.len() {
        let dt = (samples[i].0 - samples[i - 1].0) as f64 / 1e6;
        if dt > 0.0 { rate[i] = (samples[i - 1].1.inverse() * samples[i].1).scaled_axis() / dt; }
    }
    let radius = match region.mode { RepairMode::BurstBridge => 30_000, _ => 70_000 };
    let mut baseline = vec![Vector3::zeros(); samples.len()];
    for i in first..=last {
        let lo = samples.partition_point(|x| x.0 < samples[i].0 - radius);
        let hi = samples.partition_point(|x| x.0 <= samples[i].0 + radius);
        baseline[i] = Vector3::from_fn(|axis, _| median(&mut rate[lo.max(1)..hi].iter().map(|v| v[axis]).collect::<Vec<_>>()));
    }
    let mut departures: Vec<f64> = (first + 1..=last).map(|i| (rate[i] - baseline[i]).norm()).collect();
    let typical = median(&mut departures).max(0.002);
    let ceiling = match region.mode { RepairMode::BurstBridge => typical * 7.0, _ => typical * 2.5 };
    if region.mode == RepairMode::BurstBridge {
        let mut i = first + 1;
        while i <= last {
            if (rate[i] - baseline[i]).norm() <= ceiling { i += 1; continue; }
            let begin = i;
            while i <= last && (rate[i] - baseline[i]).norm() > ceiling * 0.5 { i += 1; }
            let finish = i.min(last);
            if finish - begin > 100 || samples[finish].0 - samples[begin].0 > 120_000 { continue; }
            let a = begin.saturating_sub(1);
            let b = finish;
            for j in begin..b {
                let u = (samples[j].0 - samples[a].0) as f64 / (samples[b].0 - samples[a].0) as f64;
                let target = samples[a].1.slerp(&samples[b].1, u);
                let blend = region.strength * smooth_edge(samples[j].0, start, end);
                if let Some(q) = output.get_mut(&samples[j].0) { *q = samples[j].1.slerp(&target, blend); }
            }
        }
        return;
    }
    // Integrate changed rates from the first original orientation. The endpoint drift is
    // distributed across the region so the next untouched sample remains continuous.
    let mut repaired = vec![samples[first].1; last - first + 1];
    for i in first + 1..=last {
        let residual = rate[i] - baseline[i];
        let magnitude = residual.norm();
        let gain = if magnitude > ceiling { ceiling / magnitude } else { 1.0 };
        let adjusted = match region.mode {
            RepairMode::RateAttenuation => baseline[i] + residual * gain,
            RepairMode::SelectiveResidual => baseline[i] + residual * (0.25 + 0.75 * gain),
            _ => rate[i],
        };
        let dt = (samples[i].0 - samples[i - 1].0) as f64 / 1e6;
        repaired[i - first] = repaired[i - first - 1] * Quat64::from_scaled_axis(adjusted * dt);
    }
    let end_correction = repaired[last - first].inverse() * samples[last].1;
    let span = (samples[last].0 - samples[first].0) as f64;
    for i in first..=last {
        let u = (samples[i].0 - samples[first].0) as f64 / span;
        let target = repaired[i - first] * Quat64::identity().slerp(&end_correction, u);
        let blend = region.strength * smooth_edge(samples[i].0, start, end);
        if let Some(q) = output.get_mut(&samples[i].0) { *q = samples[i].1.slerp(&target, blend); }
    }
}

/// Noise-based hardware-family estimate, never a chip identity readout.
pub fn estimate_gyro(quats: &TimeQuat) -> serde_json::Value {
    let samples: Vec<_> = quats.iter().map(|(t, q)| (*t, *q)).collect();
    let unknown = || serde_json::json!({"label":"Unknown", "reason":"Insufficient quiet data"});
    if samples.len() < 1000 { return unknown(); }
    let duration = (samples.last().unwrap().0 - samples[0].0) as f64 / 1e6;
    if duration <= 0.0 { return unknown(); }
    let fs = (samples.len() - 1) as f64 / duration;
    if fs < 300.0 || fs > 5000.0 { return unknown(); }
    let mut rates = Vec::with_capacity(samples.len() - 1);
    for pair in samples.windows(2) {
        let dt = (pair[1].0 - pair[0].0) as f64 / 1e6;
        if dt <= 0.0 { continue; }
        rates.push((pair[0].1.inverse() * pair[1].1).scaled_axis() * (180.0 / std::f64::consts::PI / dt));
    }
    // Decide whether the camera is quiet from slow motion, not raw sensor noise.
    let smooth_n = (fs * 0.02).round().max(1.0) as usize;
    let mut smoothed = Vec::with_capacity(rates.len());
    let mut running = nalgebra::Vector3::zeros();
    for (i, value) in rates.iter().enumerate() {
        running += value;
        if i >= smooth_n { running -= rates[i - smooth_n]; }
        smoothed.push(running / (i + 1).min(smooth_n) as f64);
    }
    let win = (fs * 0.5) as usize;
    // Cap the number of spectral windows; this estimate is called from the UI.
    let step = ((fs * 0.1) as usize).max(rates.len() / 96).max(1);
    if win < 256 || step == 0 { return unknown(); }
    let fft = FftPlanner::<f64>::new().plan_fft_forward(256);
    let mut windows = Vec::new();
    for offset in (0..rates.len().saturating_sub(win)).step_by(step) {
        let slice = &rates[offset..offset + win];
        let slow = &smoothed[offset..offset + win];
        let avg = slow.iter().fold(nalgebra::Vector3::zeros(), |a, x| a + x) / win as f64;
        let motion = (slow.iter().map(|x| (x - avg).norm_squared()).sum::<f64>() / win as f64).sqrt();
        if motion > 0.5 { continue; }
        let mut axis_noise = Vec::new();
        for axis in 0..3 {
            let mut psd = Vec::new();
            for block in slice.chunks_exact(256) {
                let mean = block.iter().map(|v| v[axis]).sum::<f64>() / 256.0;
                let mut data: Vec<_> = block.iter().enumerate().map(|(i, v)| {
                    let w = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / 255.0).cos();
                    Complex::new((v[axis] - mean) * w, 0.0)
                }).collect();
                fft.process(&mut data);
                for k in 1..128 {
                    let hz = k as f64 * fs / 256.0;
                    if (40.0..=130.0).contains(&hz) {
                        psd.push((data[k].norm_sqr() * 2.0 / (fs * 95.625)).sqrt() * 1000.0);
                    }
                }
            }
            axis_noise.push(median(&mut psd));
        }
        windows.push(axis_noise.iter().sum::<f64>() / 3.0);
    }
    if windows.len() < 3 { return unknown(); }
    windows.sort_by(f64::total_cmp);
    let quiet_count = (windows.len() / 4).max(1);
    let noise = median(&mut windows[..quiet_count]);
    let label = if (2.5..=5.5).contains(&noise) { "1469D / ICM-40609-D" }
        else if (6.0..=40.0).contains(&noise) { "MP66 / MPU-6600" }
        else { "Unknown" };
    serde_json::json!({"label":label, "noise":noise})
}

#[cfg(test)]
mod tests {
    use super::*;

    fn linear_track() -> TimeQuat {
        (0..100).map(|i| (i * 10_000, Quat64::from_euler_angles(0.0, i as f64 * 0.001, 0.0))).collect()
    }

    #[test]
    fn clean_rotation_has_no_candidate() {
        assert!(detect(&linear_track(), 1000.0).is_empty());
    }

    #[test]
    fn flatten_keeps_motion_outside_region_and_blends_inside() {
        let mut track = linear_track();
        track.insert(500_000, Quat64::from_euler_angles(0.0, 0.8, 0.0));
        let before = track.clone();
        apply(&mut track, &[RepairRegion { start_ms: 300.0, end_ms: 700.0, mode: RepairMode::Flatten, strength: 1.0 }]);
        assert_eq!(track.get(&100_000), before.get(&100_000));
        assert_eq!(track.get(&900_000), before.get(&900_000));
        assert!((track[&500_000].inverse() * before[&500_000]).angle() > 0.1);
    }

    #[test]
    fn direction_reversal_is_detected_even_when_speed_is_steady() {
        let mut track = TimeQuat::new();
        let mut yaw = 0.0;
        for i in 0..600 {
            let rate = if (300..330).contains(&i) { if i % 2 == 0 { 0.012 } else { -0.012 } } else { 0.0001 };
            yaw += rate;
            track.insert(i * 10_000, Quat64::from_euler_angles(0.0, yaw, 0.0));
        }
        let candidates = detect(&track, 6000.0);
        assert!(candidates.iter().any(|r| r.start_ms < 3100.0 && r.end_ms > 2900.0));
    }

    #[test]
    fn regions_round_trip_and_old_mode_names_load() {
        let regions = vec![RepairRegion { start_ms: 1200.0, end_ms: 1800.0, mode: RepairMode::MediumSmoothing, strength: 0.7 }];
        let json = serde_json::to_string(&regions).unwrap();
        assert_eq!(serde_json::from_str::<Vec<RepairRegion>>(&json).unwrap(), regions);
        assert_eq!(serde_json::from_str::<RepairMode>("\"doctor\"").unwrap(), RepairMode::ShortSmoothing);
        assert_eq!(serde_json::from_str::<RepairMode>("\"selective\"").unwrap(), RepairMode::MediumSmoothing);
    }

    #[test]
    fn adaptive_modes_change_a_burst_and_preserve_endpoints() {
        let mut source = linear_track();
        source.insert(500_000, Quat64::from_euler_angles(0.0, 0.8, 0.0));
        for mode in [RepairMode::RateAttenuation, RepairMode::BurstBridge, RepairMode::SelectiveResidual] {
            let mut result = source.clone();
            apply(&mut result, &[RepairRegion { start_ms: 300.0, end_ms: 700.0, mode, strength: 1.0 }]);
            assert_eq!(result.get(&100_000), source.get(&100_000));
            assert_eq!(result.get(&900_000), source.get(&900_000));
            assert!((result[&500_000].inverse() * source[&500_000]).angle() > 0.01, "{mode:?}");
        }
    }
}
