export function createWorker() {
    return new Worker(new URL("../workers/decompresser.worker.ts", import.meta.url), {
      type: "module",
    });
  }
  