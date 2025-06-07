"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { createWorker } from "@/utils/createWorker";

export default function Home() {
  const [name, setName] = useState("");
  const [isDriving, setIsDriving] = useState(false);
  const [gameOver, setGameOver] = useState(false);
  const [score, setScore] = useState<number | null>(null);
  const [countdown, setCountdown] = useState<number | null>(null);
  const [speed, setSpeed] = useState<number>(0);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wsRef = useRef<WebSocket>();
  const workerRef = useRef<Worker>();

  const DISPLAY_SCALE = 4;

  const resetGame = () => {
    setIsDriving(false);
    setGameOver(false);
    setScore(null);
    setCountdown(null);
    setSpeed(0);
    setName("");
  };

  const beginCountdown = useCallback(() => {
    let count = 3;
    setCountdown(count);

    const interval = setInterval(() => {
      count -= 1;
      if (count === 0) {
        clearInterval(interval);
        setCountdown(null);
        setIsDriving(true);
      } else {
        setCountdown(count);
      }
    }, 1000);
  }, []);

  useEffect(() => {
    if (!isDriving) return;

    const canvasEl = canvasRef.current!;
    const ws = new WebSocket("ws://localhost:8000/ws/play");
    ws.binaryType = "arraybuffer";
    wsRef.current = ws;

    const worker = createWorker();
    workerRef.current = worker;

    ws.onopen = () => {
      ws.send(JSON.stringify({ type: "start", name }));
    };

    ws.onmessage = async (event) => {
      if (typeof event.data === "string") {
        const msg = JSON.parse(event.data);

        if (msg.type === "init") {
          const { width, height } = msg;
          canvasEl.width = width;
          canvasEl.height = height;
          canvasEl.style.width = `${width * DISPLAY_SCALE}px`;
          canvasEl.style.height = `${height * DISPLAY_SCALE}px`;

          const offscreen = canvasEl.transferControlToOffscreen();
          worker.postMessage(
            { type: "init", canvas: offscreen, width, height },
            [offscreen]
          );
        }

        if (msg.type === "frame_meta") {
          setScore(msg.reward);
        }

        if (msg.type === "done") {
          setScore(msg.score ?? 0);
          setGameOver(true);
          setIsDriving(false);
          ws.close();
        }
        return;
      }

      const comp = new Uint8Array(await event.data.slice(0));
      worker.postMessage({ type: "frame", comp });
    };

    ws.onerror = (e) => console.error("WS error:", e);
    ws.onclose = () => console.warn("WS closed");

    return () => {
      ws.close();
      worker.terminate();
    };
  }, [isDriving, name]);

  const sendAction = useCallback((action: number[]) => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "action", action }));
    }
    setSpeed(action[1] * 100); // Estimate speed from gas
  }, []);

  useEffect(() => {
    if (!isDriving) return;
    const id = setInterval(() => sendAction([0, 0, 0]), 100);
    return () => clearInterval(id);
  }, [isDriving, sendAction]);

  const handleKey = useCallback(
    (e: KeyboardEvent) => {
      if (!isDriving || countdown !== null) return;
      switch (e.key) {
        case "ArrowUp":
          sendAction([0, 1, 0]);
          break;
        case "ArrowLeft":
          sendAction([-1, 0, 0]);
          break;
        case "ArrowRight":
          sendAction([1, 0, 0]);
          break;
        case "ArrowDown":
          sendAction([0, 0, 0.8]);
          break;
      }
    },
    [isDriving, countdown, sendAction]
  );

  useEffect(() => {
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [handleKey]);

  const start = useCallback(() => {
    if (!name.trim()) {
      alert("Enter your name");
      return;
    }
    beginCountdown();
  }, [name, beginCountdown]);

  const mobileBtnStyle: React.CSSProperties = {
    padding: "1rem",
    minWidth: "60px",
    minHeight: "60px",
    fontSize: "1.5rem",
    borderRadius: "0.75rem",
    border: "1px solid #ccc",
    backgroundColor: "#f0f0f0",
    touchAction: "manipulation",
  };

  return (
    <div style={{ textAlign: "center", padding: "2rem", position: "relative" }}>
      {!isDriving && !gameOver && countdown === null && (
        <>
          <h1>Enter Your Name</h1>
          <input
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="Driver name"
            style={{ padding: "0.5rem", marginRight: "1rem" }}
          />
          <button onClick={start} style={{ padding: "0.5rem 1rem" }}>
            Start Driving
          </button>
        </>
      )}

      {isDriving && (
        <>
          <h2>Welcome, {name}!</h2>
          <canvas
            ref={canvasRef}
            style={{
              border: "1px solid #ccc",
              imageRendering: "pixelated",
              display: "block",
              margin: "1rem auto",
              touchAction: "none",
            }}
          />
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              marginTop: "1rem",
              gap: "0.5rem",
            }}
          >
            <div>
              <button onClick={() => sendAction([0, 1, 0])} style={mobileBtnStyle}>⬆️</button>
            </div>
            <div style={{ display: "flex", gap: "1rem" }}>
              <button onClick={() => sendAction([-1, 0, 0])} style={mobileBtnStyle}>⬅️</button>
              <button onClick={() => sendAction([0, 0, 0.8])} style={mobileBtnStyle}>⏹️</button>
              <button onClick={() => sendAction([1, 0, 0])} style={mobileBtnStyle}>➡️</button>
            </div>
          </div>
        </>
      )}

      {gameOver && (
        <>
          <h2>🏁 Game Over!</h2>
          <p style={{ fontSize: "1.5rem" }}>Final Score: <strong>{score}</strong></p>
          <button onClick={resetGame} style={{ padding: "0.75rem 1.5rem", fontSize: "1rem", marginTop: "1rem" }}>
            🔁 Play Again
          </button>
        </>
      )}

      {isDriving && (
        <div style={{
          position: "absolute",
          top: "1rem",
          left: "1rem",
          backgroundColor: "rgba(0,0,0,0.6)",
          color: "white",
          padding: "0.5rem 1rem",
          borderRadius: "0.5rem",
          fontSize: "1rem"
        }}>
          🏆 Score: {score?.toFixed(2) ?? "0.00"}<br />
          🏎️ Speed: {speed.toFixed(0)}
        </div>
      )}

      {countdown !== null && (
        <div style={{
          position: "absolute",
          top: "40%",
          left: "50%",
          transform: "translate(-50%, -50%)",
          fontSize: "4rem",
          fontWeight: "bold",
          backgroundColor: "rgba(0, 0, 0, 0.7)",
          color: "white",
          padding: "1rem 2rem",
          borderRadius: "1rem"
        }}>
          {countdown === 0 ? "GO!" : countdown}
        </div>
      )}
    </div>
  );
}
