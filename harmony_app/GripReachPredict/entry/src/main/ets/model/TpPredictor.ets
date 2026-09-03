import { TpFeatureExtractor } from './TpFeatures';
import { RandomForestInfer, PredictResult, RfModel } from './RandomForestInfer';

export type TpPredictCallback = (result: PredictResult, modality: string) => void;

/**
 * Public HarmonyOS APIs do not expose raw capacitive TP matrices.
 * Feed first-order diff frames via pushDiffFrame() from a vendor TP callback.
 * Flat layout per frame: 34*21 grid + 55 line values.
 */
export class TpPredictor {
  private infer: RandomForestInfer | null = null;
  private frames: number[][] = [];
  private readonly maxFrames: number = 24;
  private onResult: TpPredictCallback | null = null;
  private lineLen: number = 55;
  private nFramesWindow: number = 11;

  loadModel(jsonText: string): void {
    const model = JSON.parse(jsonText) as RfModel;
    this.infer = new RandomForestInfer(model);
  }

  setCallback(cb: TpPredictCallback): void {
    this.onResult = cb;
  }

  pushDiffFrame(flatDiff: number[]): void {
    this.frames.push(flatDiff.slice());
    if (this.frames.length > this.maxFrames) {
      this.frames.shift();
    }
    this.maybePredict();
  }

  injectDemoPulse(kind: 'left_to_right' | 'right_to_left' | 'others'): void {
    const cell = TpFeatureExtractor.GRID_ROWS * TpFeatureExtractor.GRID_COLS;
    this.frames = [];
    for (let i = 0; i < this.nFramesWindow; i++) {
      const frame = new Array(cell + this.lineLen).fill(0) as number[];
      if (kind === 'left_to_right') {
        for (let r = 28; r < 34; r++) {
          for (let c = 15; c < 21; c++) {
            frame[r * 21 + c] = i > 5 ? 8 : 1;
          }
        }
      } else if (kind === 'right_to_left') {
        for (let r = 28; r < 34; r++) {
          for (let c = 0; c < 6; c++) {
            frame[r * 21 + c] = i > 5 ? 8 : 1;
          }
        }
      }
      this.pushDiffFrame(frame);
    }
  }

  clear(): void {
    this.frames = [];
  }

  private maybePredict(): void {
    if (this.infer === null || this.onResult === null) {
      return;
    }
    if (this.frames.length < this.nFramesWindow) {
      return;
    }
    const window = this.frames.slice(this.frames.length - this.nFramesWindow);
    const cell = TpFeatureExtractor.GRID_ROWS * TpFeatureExtractor.GRID_COLS;
    const gridAbs: number[][] = [];
    const lineAbs: number[][] = [];
    const gridDiff: number[][] = [];
    const lineDiff: number[][] = [];
    let g = new Array(cell).fill(0) as number[];
    let l = new Array(this.lineLen).fill(0) as number[];
    for (let f = 0; f < window.length; f++) {
      const raw = window[f];
      const gd = raw.slice(0, cell);
      const ld = raw.slice(cell, cell + this.lineLen);
      const ga: number[] = new Array(cell);
      const la: number[] = new Array(this.lineLen);
      for (let i = 0; i < cell; i++) {
        ga[i] = g[i] + gd[i];
      }
      for (let i = 0; i < this.lineLen; i++) {
        la[i] = l[i] + (i < ld.length ? ld[i] : 0);
      }
      g = ga;
      l = la;
      gridAbs.push(ga);
      lineAbs.push(la);
      gridDiff.push(gd);
      lineDiff.push(ld);
    }
    try {
      const feats = TpFeatureExtractor.fromAbsoluteWindow(gridAbs, lineAbs, gridDiff, lineDiff);
      const result = this.infer.predict(feats);
      this.onResult(result, 'tp');
    } catch (e) {
      console.error(`tp predict error ${JSON.stringify(e)}`);
    }
  }
}
