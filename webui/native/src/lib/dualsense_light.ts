// The controller's light bar as the character's eye.
//
// A DualSense takes light-bar colours over a plain HID output report, which a
// browser can write through WebHID -- no driver, no native helper. The report
// layout differs by transport: over USB it is report 0x02 with 47 payload
// bytes; over Bluetooth it is report 0x31, shifted one byte and terminated by
// a CRC32 over the report preceded by the 0xA2 HID seed.
//
// Everything here degrades to a no-op: WebHID is Chromium-only, access needs a
// user gesture, and the controller can be unplugged mid-animation. A missing
// light show must never disturb the conversation.

const SONY = 0x054c;
const DUALSENSE = 0x0ce6;
const DUALSENSE_EDGE = 0x0df2;

// valid_flag0: allow the rumble/haptic fields we leave at zero.
const FLAG0 = 0xff;
// valid_flag1: mic-LED (0x01), power-save (0x02), LIGHT BAR (0x04),
// player-indicator (0x10), plus 0x40 as the reference implementations send it.
const FLAG1 = 0x01 | 0x02 | 0x04 | 0x10 | 0x40;
const LED_UNINTERRUPTIBLE = 0x02;  // do not let the controller animate over us
const PULSE_OFF = 0x00;            // we drive every frame ourselves
const BRIGHTNESS_HIGH = 0x00;

let crcTable: Uint32Array | null = null;
function crc32(bytes: Uint8Array): number {
  if (!crcTable) {
    crcTable = new Uint32Array(256);
    for (let i = 0; i < 256; i += 1) {
      let c = i;
      for (let k = 0; k < 8; k += 1) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
      crcTable[i] = c >>> 0;
    }
  }
  let crc = 0xffffffff;
  for (const b of bytes) crc = (crcTable[(crc ^ b) & 0xff] ^ (crc >>> 8)) >>> 0;
  return (crc ^ 0xffffffff) >>> 0;
}

export type LightState = 'off' | 'listening' | 'hearing' | 'speaking';

export class DualSenseLight {
  private device: HIDDevice | null = null;
  private bluetooth = false;
  private timer: ReturnType<typeof setInterval> | null = null;
  private state: LightState = 'off';
  private phase = 0;

  get connected(): boolean {
    return this.device !== null && this.device.opened;
  }

  static get supported(): boolean {
    return typeof navigator !== 'undefined' && 'hid' in navigator;
  }

  // Must be called from a user gesture: WebHID will not prompt otherwise.
  async connect(): Promise<boolean> {
    if (!DualSenseLight.supported) return false;
    try {
      const granted = await navigator.hid.getDevices();
      let device = granted.find((d) => d.vendorId === SONY);
      if (!device) {
        const picked = await navigator.hid.requestDevice({
          filters: [
            { vendorId: SONY, productId: DUALSENSE },
            { vendorId: SONY, productId: DUALSENSE_EDGE }
          ]
        });
        device = picked[0];
      }
      if (!device) return false;
      if (!device.opened) await device.open();
      this.device = device;
      // Bluetooth exposes the big 0x31 output report; USB exposes 0x02. The
      // descriptor is the hint, but some HID stacks do not enumerate output
      // reports at all, so the real test is whether a write succeeds.
      const reports = device.collections.flatMap((c) => c.outputReports ?? []);
      this.bluetooth =
        reports.length > 0 && !reports.some((r) => r.reportId === 0x02);
      if (!(await this.probe())) {
        this.bluetooth = !this.bluetooth;
        if (!(await this.probe())) {
          this.device = null;
          return false;
        }
      }
      this.start();
      return true;
    } catch {
      this.device = null;
      return false;
    }
  }

  setState(state: LightState): void {
    if (state !== this.state) {
      this.state = state;
      this.phase = 0;
    }
  }

  private start(): void {
    if (this.timer !== null) return;
    // 20 Hz is plenty for a breathing light and stays far clear of the
    // controller's report budget.
    this.timer = setInterval(() => void this.frame(), 50);
  }

  async disconnect(): Promise<void> {
    if (this.timer !== null) {
      clearInterval(this.timer);
      this.timer = null;
    }
    this.state = 'off';
    await this.write(0, 0, 0);
    try {
      await this.device?.close();
    } catch {
      // Already gone; nothing to release.
    }
    this.device = null;
  }

  private frame(): void {
    this.phase += 0.05;
    const [r, g, b] = this.colorFor(this.state, this.phase);
    void this.write(r, g, b);
  }

  // HAL's eye is red; the states differ by how it breathes, not by hue, so a
  // glance across the room reads as one presence rather than a status light.
  private colorFor(state: LightState, phase: number): [number, number, number] {
    switch (state) {
      case 'listening': {
        // Slow breath, dim: waiting, patient.
        const t = (Math.sin(phase * 0.9) + 1) / 2;
        return [Math.round(30 + t * 70), 0, 0];
      }
      case 'hearing': {
        // Steady and bright while it takes your words in.
        const t = (Math.sin(phase * 3.0) + 1) / 2;
        return [Math.round(190 + t * 40), Math.round(10 + t * 10), 0];
      }
      case 'speaking': {
        // Full, with a slow swell under it, while it answers.
        const t = (Math.sin(phase * 1.6) + 1) / 2;
        return [255, Math.round(t * 24), Math.round(t * 16)];
      }
      default:
        return [0, 0, 0];
    }
  }

  // One write, reporting success -- used to settle the transport question
  // before the animation loop starts (where failures are silent by design).
  private async probe(): Promise<boolean> {
    const device = this.device;
    if (!device) return false;
    try {
      await this.send(device, 40, 0, 0);
      return true;
    } catch {
      return false;
    }
  }

  private async write(r: number, g: number, b: number): Promise<void> {
    const device = this.device;
    if (!device || !device.opened) return;
    try {
      await this.send(device, r, g, b);
    } catch {
      // Unplugged mid-frame, or the report was rejected: stop animating
      // rather than throwing on every tick.
      if (this.timer !== null) {
        clearInterval(this.timer);
        this.timer = null;
      }
      this.device = null;
    }
  }

  private async send(device: HIDDevice, r: number, g: number, b: number): Promise<void> {
    {
      if (this.bluetooth) {
        // Report 0x31: one leading flag byte, the USB body shifted by one,
        // then a CRC32 over the 0xA2 seed plus the whole report.
        const data = new Uint8Array(77);
        data[0] = 0x02;
        data[1] = FLAG0;
        data[2] = FLAG1;
        data[39] = LED_UNINTERRUPTIBLE;
        data[42] = PULSE_OFF;
        data[43] = BRIGHTNESS_HIGH;
        data[45] = r;
        data[46] = g;
        data[47] = b;
        const framed = new Uint8Array(1 + 1 + data.length - 4);
        framed[0] = 0xa2;
        framed[1] = 0x31;
        framed.set(data.subarray(0, data.length - 4), 2);
        const crc = crc32(framed);
        data[73] = crc & 0xff;
        data[74] = (crc >>> 8) & 0xff;
        data[75] = (crc >>> 16) & 0xff;
        data[76] = (crc >>> 24) & 0xff;
        await device.sendReport(0x31, data);
      } else {
        const data = new Uint8Array(47);
        data[0] = FLAG0;
        data[1] = FLAG1;
        data[38] = LED_UNINTERRUPTIBLE;
        data[41] = PULSE_OFF;
        data[42] = BRIGHTNESS_HIGH;
        data[44] = r;
        data[45] = g;
        data[46] = b;
        await device.sendReport(0x02, data);
      }
    }
  }
}
