import { sensor } from '@kit.SensorServiceKit';
import { BusinessError } from '@kit.BasicServicesKit';
import { ImuFeatureExtractor, Sample6 } from './ImuFeatures';
import { RandomForestInfer, PredictResult, RfModel } from './RandomForestInfer';

export type PredictCallback = (result: PredictResult, modality: string) => void;

export class ImuPredictor {
  private infer: RandomForestInfer | null = null;
  private samples: Sample6[] = [];
  private lastAx: number = 0;
  private lastAy: number = 0;
  private lastAz: number = 0;
  private lastGx: number = 0;
  private lastGy: number = 0;
  private lastGz: number = 0;
  private hasAcc: boolean = false;
  private hasGyro: boolean = false;
  private running: boolean = false;
  private timerId: number = -1;
  private onResult: PredictCallback | null = null;
  private readonly keepMs: number = 800;
  private readonly strideMs: number = 50;

  loadModel(jsonText: string): void {
    const model = JSON.parse(jsonText) as RfModel;
    this.infer = new RandomForestInfer(model);
  }

  setCallback(cb: PredictCallback): void {
    this.onResult = cb;
  }

  start(): void {
    if (this.running) {
      return;
    }
    this.running = true;
    const interval = 10000000;
    try {
      sensor.on(sensor.SensorId.ACCELEROMETER, (data: sensor.AccelerometerResponse) => {
        this.lastAx = data.x;
        this.lastAy = data.y;
        this.lastAz = data.z;
        this.hasAcc = true;
        this.pushIfReady(Number(data.timestamp));
      }, { interval: interval });
    } catch (e) {
      const err = e as BusinessError;
      console.error(`acc on failed: ${err.code} ${err.message}`);
    }
    try {
      sensor.on(sensor.SensorId.GYROSCOPE, (data: sensor.GyroscopeResponse) => {
        this.lastGx = data.x;
        this.lastGy = data.y;
        this.lastGz = data.z;
        this.hasGyro = true;
        this.pushIfReady(Number(data.timestamp));
      }, { interval: interval });
    } catch (e) {
      const err = e as BusinessError;
      console.error(`gyro on failed: ${err.code} ${err.message}`);
    }
    this.timerId = setInterval(() => {
      this.tick();
    }, this.strideMs);
  }

  stop(): void {
    this.running = false;
    try {
      sensor.off(sensor.SensorId.ACCELEROMETER);
    } catch (_e) {
    }
    try {
      sensor.off(sensor.SensorId.GYROSCOPE);
    } catch (_e) {
    }
    if (this.timerId >= 0) {
      clearInterval(this.timerId);
      this.timerId = -1;
    }
  }

  private pushIfReady(timestampNs: number): void {
    if (!this.hasAcc || !this.hasGyro) {
      return;
    }
    const tMs = timestampNs / 1e6;
    this.samples.push({
      t: tMs,
      ax: this.lastAx,
      ay: this.lastAy,
      az: this.lastAz,
      gx: this.lastGx,
      gy: this.lastGy,
      gz: this.lastGz
    });
    const cutoff = tMs - this.keepMs;
    while (this.samples.length > 2 && this.samples[0].t < cutoff) {
      this.samples.shift();
    }
  }

  private tick(): void {
    if (this.infer === null || this.samples.length < 8 || this.onResult === null) {
      return;
    }
    const tEnd = this.samples[this.samples.length - 1].t;
    const seq = ImuFeatureExtractor.resample(this.samples, tEnd, this.infer.windowMs, 100);
    if (seq === null) {
      return;
    }
    try {
      const feats = ImuFeatureExtractor.extract(seq);
      const result = this.infer.predict(feats);
      this.onResult(result, 'imu');
    } catch (e) {
      console.error(`imu predict error ${JSON.stringify(e)}`);
    }
  }
}
