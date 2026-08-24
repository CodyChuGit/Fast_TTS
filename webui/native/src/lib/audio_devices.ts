// Audio device selection for the voice loop, with one opinion: when a game
// controller with a mic and speaker is plugged in (DualSense exposes both),
// it is almost certainly the device the user wants to talk through.

const CONTROLLER_RX = /dualsense|wireless controller/i;

export interface AudioDeviceOption {
  deviceId: string;
  label: string;
  controller: boolean;
}

export interface AudioDeviceSets {
  inputs: AudioDeviceOption[];
  outputs: AudioDeviceOption[];
}

function toOption(device: MediaDeviceInfo, index: number, kind: string): AudioDeviceOption {
  const label = device.label || `${kind} ${index + 1}`;
  return { deviceId: device.deviceId, label, controller: CONTROLLER_RX.test(label) };
}

// Labels are only readable once mic permission has been granted at least
// once; before that the lists still enumerate with placeholder names.
export async function listAudioDevices(): Promise<AudioDeviceSets> {
  if (!navigator.mediaDevices?.enumerateDevices) return { inputs: [], outputs: [] };
  const devices = await navigator.mediaDevices.enumerateDevices();
  return {
    inputs: devices
      .filter((d) => d.kind === 'audioinput' && d.deviceId !== 'communications')
      .map((d, i) => toOption(d, i, 'Microphone')),
    outputs: devices
      .filter((d) => d.kind === 'audiooutput' && d.deviceId !== 'communications')
      .map((d, i) => toOption(d, i, 'Speaker'))
  };
}

// The auto-pick: an explicit saved choice wins; otherwise the controller if
// present; otherwise the browser default (empty id).
export function preferredDevice(
  options: AudioDeviceOption[],
  savedId: string | null
): string {
  if (savedId && options.some((o) => o.deviceId === savedId)) return savedId;
  const controller = options.find((o) => o.controller);
  return controller ? controller.deviceId : '';
}

const INPUT_KEY = 'voice.input_device';
const OUTPUT_KEY = 'voice.output_device';

export function savedInputId(): string | null {
  try { return localStorage.getItem(INPUT_KEY); } catch { return null; }
}
export function savedOutputId(): string | null {
  try { return localStorage.getItem(OUTPUT_KEY); } catch { return null; }
}
export function rememberDevices(inputId: string, outputId: string): void {
  try {
    localStorage.setItem(INPUT_KEY, inputId);
    localStorage.setItem(OUTPUT_KEY, outputId);
  } catch {
    // Storage may be unavailable; selection still applies for this session.
  }
}
