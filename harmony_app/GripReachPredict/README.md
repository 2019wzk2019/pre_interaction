# GripReachPredict — HarmonyOS HAP

三分类预测：`左to右`（左手够右下角）/ `右to左`（右手够左下角）/ `others`。

## 交互

| 预测 | 屏幕 |
|------|------|
| 左to右 | **右半屏** 亮绿灯 3s |
| 右to左 | **左半屏** 亮绿灯 3s |
| others | 黑屏 |

可在界面切换 **IMU 模型** / **TP 模型**。

## 在 DevEco Studio 打开

1. 安装 [DevEco Studio](https://developer.huawei.com/consumer/cn/deveco-studio/)（API 12 / HarmonyOS NEXT）。
2. **Open** 本目录：`harmony_app/GripReachPredict`
3. 配置签名（自动签名即可），连接真机或模拟器。
4. Run `entry` 生成并安装 HAP。

重新导出模型（可选）：

```bash
python export_harmony_models.py
```

模型写入 `entry/src/main/resources/rawfile/{imu,tp}_model.json`。

## 模态说明

### IMU（推荐真机直接用）

- 使用系统加速度计 + 陀螺仪。
- 150ms 窗、与训练一致的手工特征 + 导出的 RandomForest。
- 需要权限：`ohos.permission.ACCELEROMETER`、`ohos.permission.GYROSCOPE`。

### TP

- **公开 HarmonyOS API 读不到训练时那种原始电容差分矩阵**（你们采集 App 里的 `diffdata`）。
- 工程里保留了完整 TP 推理：`TpPredictor.pushDiffFrame(flat)`，接厂商/系统 TP 回调即可。
- 界面提供「演示:左→右 / 右→左」脉冲，只用于验证绿灯 UI；**演示脉冲不是真实模型输入质量保证**。

## 工程结构

```
harmony_app/GripReachPredict/
  AppScope/
  entry/src/main/ets/
    pages/Index.ets          # UI + 半屏绿灯
    model/RandomForestInfer.ets
    model/ImuFeatures.ets
    model/ImuPredictor.ets
    model/TpFeatures.ets
    model/TpPredictor.ets
  entry/src/main/resources/rawfile/
    imu_model.json
    tp_model.json
```

## 注意

- 手机 slim 模型：IMU RF(80)、TP RF(60)，测试集准确率仍接近完整模型。
- 训练数据仍是「同一次录音切分」，跨设备泛化需更多采集。
