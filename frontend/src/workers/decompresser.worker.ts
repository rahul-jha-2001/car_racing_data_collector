import pako from "pako";

self.onmessage = (e) => {
  const { comp, width, height } = e.data;

  try {
    const raw = pako.inflate(comp);

    const rgba = new Uint8ClampedArray(width * height * 4);
    for (let i = 0, j = 0; i < raw.length; i += 3, j += 4) {
      rgba[j] = raw[i];
      rgba[j + 1] = raw[i + 1];
      rgba[j + 2] = raw[i + 2];
      rgba[j + 3] = 255;
    }

    self.postMessage({ rgba, width, height });
  } catch (err) {
    self.postMessage({ error: String(err) });
  }
};

export default null as any; // 👈 required for TypeScript compatibility
