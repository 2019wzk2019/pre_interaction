/** IMU 150ms window features — mirrors grip_classify.features.imu_window_features */

export interface Sample6 {
  t: number;
  ax: number;
  ay: number;
  az: number;
  gx: number;
  gy: number;
  gz: number;
}

export class ImuFeatureExtractor {
  static extract(seq: number[][]): number[] {
    const T = seq.length;
    const C = 6;
    const feats: number[] = [];
    const last = seq[T - 1];
    const first = seq[0];
    for (let i = 0; i < C; i++) {
      feats.push(last[i]);
    }
    for (let i = 0; i < C; i++) {
      feats.push(first[i]);
    }
    for (let i = 0; i < C; i++) {
      feats.push(last[i] - first[i]);
    }
    for (let c = 0; c < C; c++) {
      let sum = 0;
      let sumSq = 0;
      let sumAbs = 0;
      let mn = seq[0][c];
      let mx = seq[0][c];
      for (let t = 0; t < T; t++) {
        const v = seq[t][c];
        sum += v;
        sumSq += v * v;
        sumAbs += Math.abs(v);
        if (v < mn) {
          mn = v;
        }
        if (v > mx) {
          mx = v;
        }
      }
      const mean = sum / T;
      let varSum = 0;
      for (let t = 0; t < T; t++) {
        const d = seq[t][c] - mean;
        varSum += d * d;
      }
      const std = Math.sqrt(varSum / T);
      feats.push(mean, std, mn, mx, mx - mn, sumSq / T, sumAbs / T);
    }

    const accMag: number[] = [];
    const gyroMag: number[] = [];
    for (let t = 0; t < T; t++) {
      const a = seq[t];
      accMag.push(Math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]));
      gyroMag.push(Math.sqrt(a[3] * a[3] + a[4] * a[4] + a[5] * a[5]));
    }
    for (const mag of [accMag, gyroMag]) {
      let sum = 0;
      let mn = mag[0];
      let mx = mag[0];
      for (let i = 0; i < mag.length; i++) {
        sum += mag[i];
        if (mag[i] < mn) {
          mn = mag[i];
        }
        if (mag[i] > mx) {
          mx = mag[i];
        }
      }
      const mean = sum / mag.length;
      let varSum = 0;
      for (let i = 0; i < mag.length; i++) {
        const d = mag[i] - mean;
        varSum += d * d;
      }
      const std = Math.sqrt(varSum / mag.length);
      let diffAbs = 0;
      for (let i = 1; i < mag.length; i++) {
        diffAbs += Math.abs(mag[i] - mag[i - 1]);
      }
      const meanAbsDiff = mag.length > 1 ? diffAbs / (mag.length - 1) : 0;
      feats.push(mean, std, mx, meanAbsDiff);
    }

    let gx = 0;
    let gy = 0;
    let gz = 0;
    for (let t = 0; t < T; t++) {
      gx += seq[t][0];
      gy += seq[t][1];
      gz += seq[t][2];
    }
    gx /= T;
    gy /= T;
    gz /= T;
    const gnorm = Math.sqrt(gx * gx + gy * gy + gz * gz) + 1e-8;
    const ghat0 = gx / gnorm;
    const ghat1 = gy / gnorm;
    const ghat2 = gz / gnorm;
    feats.push(ghat0, ghat1, ghat2);
    feats.push(Math.atan2(ghat1, ghat0));
    feats.push(Math.atan2(ghat2, Math.sqrt(ghat0 * ghat0 + ghat1 * ghat1)));

    let mgx = 0;
    let mgy = 0;
    let mgz = 0;
    for (let t = 0; t < T; t++) {
      mgx += seq[t][3];
      mgy += seq[t][4];
      mgz += seq[t][5];
    }
    feats.push(mgx / T, mgy / T, mgz / T);

    if (T > 2) {
      const jerks: number[] = [];
      for (let t = 1; t < T; t++) {
        const dx = seq[t][0] - seq[t - 1][0];
        const dy = seq[t][1] - seq[t - 1][1];
        const dz = seq[t][2] - seq[t - 1][2];
        jerks.push(Math.sqrt(dx * dx + dy * dy + dz * dz));
      }
      let jSum = 0;
      for (let i = 0; i < jerks.length; i++) {
        jSum += jerks[i];
      }
      const jMean = jSum / jerks.length;
      let jVar = 0;
      for (let i = 0; i < jerks.length; i++) {
        const d = jerks[i] - jMean;
        jVar += d * d;
      }
      feats.push(jMean, Math.sqrt(jVar / jerks.length));
    } else {
      feats.push(0, 0);
    }
    return feats;
  }

  static resample(samples: Sample6[], tEndMs: number, windowMs: number = 150, rateHz: number = 100):
    number[][] | null {
    if (samples.length < 2) {
      return null;
    }
    const t0 = tEndMs - windowMs;
    if (samples[0].t > t0 || samples[samples.length - 1].t < tEndMs) {
      return null;
    }
    const n = Math.max(2, Math.round(windowMs / 1000 * rateHz));
    const out: number[][] = [];
    for (let i = 0; i < n; i++) {
      const t = t0 + (tEndMs - t0) * (i / (n - 1));
      out.push(ImuFeatureExtractor.interpAt(samples, t));
    }
    return out;
  }

  private static interpAt(samples: Sample6[], t: number): number[] {
    if (t <= samples[0].t) {
      const s = samples[0];
      return [s.ax, s.ay, s.az, s.gx, s.gy, s.gz];
    }
    const last = samples[samples.length - 1];
    if (t >= last.t) {
      return [last.ax, last.ay, last.az, last.gx, last.gy, last.gz];
    }
    let lo = 0;
    let hi = samples.length - 1;
    while (lo + 1 < hi) {
      const mid = (lo + hi) >> 1;
      if (samples[mid].t <= t) {
        lo = mid;
      } else {
        hi = mid;
      }
    }
    const a = samples[lo];
    const b = samples[hi];
    const w = (t - a.t) / Math.max(1e-6, b.t - a.t);
    return [
      a.ax + (b.ax - a.ax) * w,
      a.ay + (b.ay - a.ay) * w,
      a.az + (b.az - a.az) * w,
      a.gx + (b.gx - a.gx) * w,
      a.gy + (b.gy - a.gy) * w,
      a.gz + (b.gz - a.gz) * w
    ];
  }
}
