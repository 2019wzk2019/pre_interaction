import { common } from '@kit.AbilityKit';
import { PredictResult } from '../model/RandomForestInfer';
import { loadRawText } from '../model/ModelLoader';
import { ImuPredictor } from '../model/ImuPredictor';
import { TpPredictor } from '../model/TpPredictor';

type Modality = 'imu' | 'tp';
type FlashSide = 'none' | 'left' | 'right';

@Entry
@Component
struct Index {
  @State modality: Modality = 'imu';
  @State running: boolean = false;
  @State statusText: string = '未启动';
  @State labelText: string = 'others';
  @State probaText: string = '';
  @State flashSide: FlashSide = 'none';
  @State tipText: string =
    'IMU：加速度计+陀螺仪实时预测。TP：系统无公开原始电容差分，需厂商回调；可用演示按钮测绿灯。';

  private imuPredictor: ImuPredictor = new ImuPredictor();
  private tpPredictor: TpPredictor = new TpPredictor();
  private modelsReady: boolean = false;

  aboutToAppear(): void {
    this.bootstrap();
  }

  aboutToDisappear(): void {
    this.stopAll();
  }

  private async bootstrap(): Promise<void> {
    try {
      const ctx = getContext(this) as common.UIAbilityContext;
      const rm = ctx.resourceManager;
      const imuJson = await loadRawText(rm, 'imu_model.json');
      const tpJson = await loadRawText(rm, 'tp_model.json');
      this.imuPredictor.loadModel(imuJson);
      this.tpPredictor.loadModel(tpJson);
      this.imuPredictor.setCallback((r: PredictResult, m: string) => {
        this.onResult(r, m);
      });
      this.tpPredictor.setCallback((r: PredictResult, m: string) => {
        this.onResult(r, m);
      });
      this.modelsReady = true;
      this.statusText = '模型已加载，可点击开始';
    } catch (e) {
      this.statusText = `模型加载失败: ${JSON.stringify(e)}`;
    }
  }

  private onResult(result: PredictResult, modality: string): void {
    if (!this.running) {
      return;
    }
    this.labelText = result.label;
    const parts: string[] = [];
    const keys = Object.keys(result.proba);
    for (let i = 0; i < keys.length; i++) {
      const k = keys[i];
      parts.push(`${k}=${result.proba[k].toFixed(2)}`);
    }
    this.probaText = parts.join('  ');
    this.statusText = `${modality.toUpperCase()} 预测中`;

    // 左to右 = 左手够右下角 → 右半屏绿灯
    // 右to左 = 右手够左下角 → 左半屏绿灯
    if (result.label.indexOf('左') === 0 && result.label.indexOf('右') > 0) {
      this.flashSide = 'right';
    } else if (result.label.indexOf('右') === 0 && result.label.indexOf('左') > 0) {
      this.flashSide = 'left';
    } else {
      this.flashSide = 'none';
    }
  }

  private stopAll(): void {
    this.running = false;
    this.imuPredictor.stop();
    this.tpPredictor.clear();
    this.flashSide = 'none';
  }

  private start(): void {
    if (!this.modelsReady) {
      this.statusText = '模型尚未就绪';
      return;
    }
    this.stopAll();
    this.running = true;
    this.labelText = 'others';
    this.probaText = '';
    if (this.modality === 'imu') {
      this.statusText = 'IMU 采集中… 请做够角动作';
      this.imuPredictor.start();
    } else {
      this.statusText = 'TP 模式：等待差分帧 / 可用演示按钮';
    }
  }

  build() {
    Stack() {
      Row() {
        Column()
          .width('50%')
          .height('100%')
          .backgroundColor(this.flashSide === 'left' ? '#22C55E' : '#000000')
        Column()
          .width('50%')
          .height('100%')
          .backgroundColor(this.flashSide === 'right' ? '#22C55E' : '#000000')
      }
      .width('100%')
      .height('100%')

      Column({ space: 12 }) {
        Text('GripReach 三分类')
          .fontSize(22)
          .fontColor(Color.White)
          .fontWeight(FontWeight.Bold)

        Text(this.tipText)
          .fontSize(12)
          .fontColor('#CBD5E1')
          .width('90%')

        Row({ space: 10 }) {
          Button(this.modality === 'imu' ? '模型: IMU ✓' : '切到 IMU')
            .backgroundColor(this.modality === 'imu' ? '#2563EB' : '#334155')
            .onClick(() => {
              this.modality = 'imu';
              if (this.running) {
                this.start();
              }
            })
          Button(this.modality === 'tp' ? '模型: TP ✓' : '切到 TP')
            .backgroundColor(this.modality === 'tp' ? '#2563EB' : '#334155')
            .onClick(() => {
              this.modality = 'tp';
              if (this.running) {
                this.start();
              }
            })
        }

        Row({ space: 10 }) {
          Button(this.running ? '停止' : '开始预测')
            .backgroundColor(this.running ? '#DC2626' : '#16A34A')
            .onClick(() => {
              if (this.running) {
                this.stopAll();
                this.statusText = '已停止';
              } else {
                this.start();
              }
            })
          if (this.modality === 'tp') {
            Button('演示:左→右')
              .onClick(() => {
                if (!this.running) {
                  this.start();
                }
                this.tpPredictor.injectDemoPulse('left_to_right');
              })
            Button('演示:右→左')
              .onClick(() => {
                if (!this.running) {
                  this.start();
                }
                this.tpPredictor.injectDemoPulse('right_to_left');
              })
          }
        }

        Text(`状态: ${this.statusText}`)
          .fontColor('#E2E8F0')
          .fontSize(14)
        Text(`预测: ${this.labelText}`)
          .fontColor(Color.White)
          .fontSize(28)
          .fontWeight(FontWeight.Medium)
        Text(this.probaText)
          .fontColor('#94A3B8')
          .fontSize(12)
        if (this.flashSide !== 'none') {
          Text(`${this.flashSide === 'right' ? '右半屏=左手够右下' : '左半屏=右手够左下'}`)
            .fontColor('#86EFAC')
            .fontSize(13)
        }
      }
      .width('100%')
      .height('100%')
      .padding(20)
      .justifyContent(FlexAlign.Center)
      .alignItems(HorizontalAlign.Center)
    }
    .width('100%')
    .height('100%')
    .backgroundColor(Color.Black)
  }
}
