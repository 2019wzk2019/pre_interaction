/** TP window features — long-edge 3 layers only, mirrors grip_classify.features.tp_window_features */

export class TpFeatureExtractor {
  static readonly GRID_ROWS: number = 34;
  static readonly GRID_COLS: number = 21;
  static readonly EDGE_LAYERS: number = 3;

  static extractFromDiffWindow(framesDiff: number[][], lineLen: number = 55): number[] | null {
    const F = framesDiff.length;
    if (F < 3) {
      return null;
    }
    const cell = TpFeatureExtractor.GRID_ROWS * TpFeatureExtractor.GRID_COLS;
    const gridAbs: number[][] = [];
    const gridDiff: number[][] = [];
    let gPrev: number[] = new Array(cell).fill(0);
    for (let f = 0; f < F; f++) {
      const gd = framesDiff[f].slice(0, cell);
      const ga: number[] = new Array(cell);
      for (let i = 0; i < cell; i++) {
        ga[i] = gPrev[i] + gd[i];
      }
      gPrev = ga;
      gridAbs.push(ga);
      gridDiff.push(gd);
    }
    const dummy: number[][] = [];
    return TpFeatureExtractor.fromAbsoluteWindow(gridAbs, dummy, gridDiff, dummy);
  }

  static fromAbsoluteWindow(
    gridAbs: number[][],
    _lineAbs: number[][],
    gridDiff: number[][],
    _lineDiff: number[][]
  ): number[] {
    const F = gridAbs.length;
    const edgeAbs: number[][] = [];
    const edgeDiff: number[][] = [];
    for (let f = 0; f < F; f++) {
      edgeAbs.push(TpFeatureExtractor.longEdge(gridAbs[f]));
      edgeDiff.push(TpFeatureExtractor.longEdge(gridDiff[f]));
    }
    const E = edgeAbs[0].length;
    const gMean: number[] = new Array(E).fill(0);
    for (let f = 0; f < F; f++) {
      for (let i = 0; i < E; i++) {
        gMean[i] += edgeAbs[f][i];
      }
    }
    for (let i = 0; i < E; i++) {
      gMean[i] /= F;
    }

    const feats: number[] = [];
    for (let i = 0; i < E; i++) {
      feats.push(edgeAbs[F - 1][i] - gMean[i]);
    }
    let globalMean = 0;
    for (let i = 0; i < E; i++) {
      globalMean += gMean[i];
    }
    globalMean /= E;
    for (let i = 0; i < E; i++) {
      feats.push(gMean[i] - globalMean);
    }
    const gStd: number[] = new Array(E).fill(0);
    for (let f = 0; f < F; f++) {
      for (let i = 0; i < E; i++) {
        const d = edgeAbs[f][i] - gMean[i];
        gStd[i] += d * d;
      }
    }
    for (let i = 0; i < E; i++) {
      feats.push(Math.sqrt(gStd[i] / F));
    }
    for (let i = 0; i < E; i++) {
      feats.push(edgeDiff[F - 1][i]);
    }
    let absG = 0;
    let cntG = 0;
    for (let f = 0; f < F; f++) {
      for (let i = 0; i < E; i++) {
        absG += Math.abs(edgeDiff[f][i]);
        cntG++;
      }
    }
    feats.push(absG / Math.max(1, cntG));
    for (let i = 0; i < E; i++) {
      feats.push(edgeAbs[F - 1][i] - edgeAbs[0][i]);
    }
    return feats;
  }

  /** Left 3 cols then right 3 cols, row-major. */
  static longEdge(flat: number[]): number[] {
    const rows = TpFeatureExtractor.GRID_ROWS;
    const cols = TpFeatureExtractor.GRID_COLS;
    const layers = TpFeatureExtractor.EDGE_LAYERS;
    const out: number[] = new Array(rows * layers * 2);
    let k = 0;
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < layers; c++) {
        out[k++] = flat[r * cols + c];
      }
      for (let c = cols - layers; c < cols; c++) {
        out[k++] = flat[r * cols + c];
      }
    }
    return out;
  }
}
