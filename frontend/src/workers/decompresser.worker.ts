import pako from "pako";

let ctx: OffscreenCanvasRenderingContext2D | null = null;
let width = 0;
let height = 0;

self.onmessage = (e) => {
  const { type } = e.data;

  if (type === "init") {
    const canvas = e.data.canvas as OffscreenCanvas;
    width = e.data.width;
    height = e.data.height;
    ctx = canvas.getContext("2d");
  }

  if (type === "frame") {
    const { comp } = e.data;

    try {
      const raw = pako.inflate(comp);
      const rgba = new Uint8ClampedArray(width * height * 4);
      for (let i = 0, j = 0; i < raw.length; i += 3, j += 4) {
        rgba[j] = raw[i];
        rgba[j + 1] = raw[i + 1];
        rgba[j + 2] = raw[i + 2];
        rgba[j + 3] = 255;
      }

      const imageData = new ImageData(rgba, width, height);
      ctx?.putImageData(imageData, 0, 0);
    } catch (err) {
      self.postMessage({ error: String(err) });
    }
  }
};
