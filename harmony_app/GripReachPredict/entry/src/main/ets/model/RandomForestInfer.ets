/** Compact RandomForest inference for exported imu_model.json / tp_model.json */

export interface RfTree {
  children_left: number[];
  children_right: number[];
  feature: number[];
  threshold: number[];
  value: number[][];
}

export interface RfModel {
  modality: string;
  labels: string[];
  window_ms: number;
  feature_dim: number;
  n_estimators: number;
  scaler_mean: number[];
  scaler_scale: number[];
  trees: RfTree[];
}

export interface PredictResult {
  label: string;
  labelId: number;
  proba: Record<string, number>;
}

export class RandomForestInfer {
  private model: RfModel;

  constructor(model: RfModel) {
    this.model = model;
  }

  get labels(): string[] {
    return this.model.labels;
  }

  get windowMs(): number {
    return this.model.window_ms;
  }

  private scale(x: number[]): number[] {
    const mean = this.model.scaler_mean;
    const scale = this.model.scaler_scale;
    const out: number[] = new Array(x.length);
    for (let i = 0; i < x.length; i++) {
      const s = (scale[i] === 0) ? 1 : scale[i];
      out[i] = (x[i] - mean[i]) / s;
    }
    return out;
  }

  private predictTree(tree: RfTree, x: number[]): number[] {
    let node = 0;
    while (true) {
      const left = tree.children_left[node];
      const right = tree.children_right[node];
      if (left < 0 || right < 0) {
        return tree.value[node];
      }
      if (x[tree.feature[node]] <= tree.threshold[node]) {
        node = left;
      } else {
        node = right;
      }
    }
  }

  predict(features: number[]): PredictResult {
    if (features.length !== this.model.feature_dim) {
      throw new Error(`feature dim ${features.length} != ${this.model.feature_dim}`);
    }
    const x = this.scale(features);
    const nClass = this.model.labels.length;
    const acc: number[] = new Array(nClass).fill(0);
    for (let t = 0; t < this.model.trees.length; t++) {
      const p = this.predictTree(this.model.trees[t], x);
      for (let c = 0; c < nClass; c++) {
        acc[c] += p[c];
      }
    }
    const nTrees = Math.max(1, this.model.trees.length);
    let best = 0;
    let bestScore = -1;
    const proba: Record<string, number> = {};
    for (let c = 0; c < nClass; c++) {
      const v = acc[c] / nTrees;
      proba[this.model.labels[c]] = v;
      if (v > bestScore) {
        bestScore = v;
        best = c;
      }
    }
    return {
      label: this.model.labels[best],
      labelId: best,
      proba: proba
    };
  }
}
