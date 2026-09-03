import { util } from '@kit.ArkTS';
import { resourceManager } from '@kit.LocalizationKit';

export async function loadRawText(rm: resourceManager.ResourceManager, name: string): Promise<string> {
  const buf = await rm.getRawFileContent(name);
  const decoder = util.TextDecoder.create('utf-8', { ignoreBOM: true });
  return decoder.decodeWithStream(buf);
}
