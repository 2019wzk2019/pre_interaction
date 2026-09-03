import { AbilityConstant, ConfigurationConstant, UIAbility, Want } from '@kit.AbilityKit';
import { hilog } from '@kit.PerformanceAnalysisKit';
import { window } from '@kit.ArkUI';

const DOMAIN = 0x3200;

export default class EntryAbility extends UIAbility {
  onCreate(want: Want, launchParam: AbilityConstant.LaunchParam): void {
    this.context.getApplicationContext().setColorMode(ConfigurationConstant.ColorMode.COLOR_MODE_NOT_SET);
    hilog.info(DOMAIN, 'GripReach', 'Ability onCreate');
  }

  onDestroy(): void {
    hilog.info(DOMAIN, 'GripReach', 'Ability onDestroy');
  }

  onWindowStageCreate(windowStage: window.WindowStage): void {
    hilog.info(DOMAIN, 'GripReach', 'Ability onWindowStageCreate');
    windowStage.loadContent('pages/Index', (err) => {
      if (err.code) {
        hilog.error(DOMAIN, 'GripReach', 'Failed to load content %{public}d', err.code);
        return;
      }
      hilog.info(DOMAIN, 'GripReach', 'Succeeded in loading content');
    });
  }

  onWindowStageDestroy(): void {
    hilog.info(DOMAIN, 'GripReach', 'Ability onWindowStageDestroy');
  }

  onForeground(): void {
    hilog.info(DOMAIN, 'GripReach', 'Ability onForeground');
  }

  onBackground(): void {
    hilog.info(DOMAIN, 'GripReach', 'Ability onBackground');
  }
}
