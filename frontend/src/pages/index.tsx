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
  const [topScores, setTopScores] = useState<
    { player: string; score: number; session_id: string }[]
  >([]);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const workerRef = useRef<Worker | null>(null);
  
  // Track pressed keys
  const keysPressed = useRef<Set<string>>(new Set());
  const actionIntervalRef = useRef<NodeJS.Timeout | null>(null);

  const DISPLAY_SCALE = 4;
  console.log(process.env.NEXT_PUBLIC_API_BASE_URL);

  const resetGame = () => {
    window.location.reload(); // 🔁 Full page refresh
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
    const ws = new WebSocket(`ws://${process.env.NEXT_PUBLIC_API_BASE_URL}/ws/play`);
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

  useEffect(() => {
    const fetchScores = async () => {
      try {
        const res = await fetch(`http://${process.env.NEXT_PUBLIC_API_BASE_URL}/api/scores`);
        const data = await res.json();
        setTopScores(data);
      } catch (err) {
        console.error("Failed to load top scores:", err);
      }
    };

    fetchScores();
  }, []);

  const sendAction = useCallback((action: number[]) => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "action", action }));
    }
    setSpeed(action[1] * 100); // Estimate speed from gas
  }, []);

  // Calculate current action based on pressed keys
  const getCurrentAction = useCallback((): number[] => {
    let steering = 0;
    let gas = 0;
    let brake = 0;

    if (keysPressed.current.has("ArrowLeft")) steering -= 1;
    if (keysPressed.current.has("ArrowRight")) steering += 1;
    if (keysPressed.current.has("ArrowUp")) gas = 1;
    if (keysPressed.current.has("ArrowDown")) brake = 0.8;

    // Alternative keys for WASD users
    if (keysPressed.current.has("a") || keysPressed.current.has("A")) steering -= 1;
    if (keysPressed.current.has("d") || keysPressed.current.has("D")) steering += 1;
    if (keysPressed.current.has("w") || keysPressed.current.has("W")) gas = 1;
    if (keysPressed.current.has("s") || keysPressed.current.has("S")) brake = 0.8;

    // Clamp steering to -1, 1 range
    steering = Math.max(-1, Math.min(1, steering));

    return [steering, gas, brake];
  }, []);

  // Continuous action sending based on held keys
  useEffect(() => {
    if (!isDriving || countdown !== null) {
      if (actionIntervalRef.current) {
        clearInterval(actionIntervalRef.current);
        actionIntervalRef.current = null;
      }
      return;
    }

    actionIntervalRef.current = setInterval(() => {
      const action = getCurrentAction();
      sendAction(action);
    }, 50); // Send actions every 50ms for smooth control

    return () => {
      if (actionIntervalRef.current) {
        clearInterval(actionIntervalRef.current);
        actionIntervalRef.current = null;
      }
    };
  }, [isDriving, countdown, getCurrentAction, sendAction]);

  const handleKeyDown = useCallback((e: KeyboardEvent) => {
    if (!isDriving || countdown !== null) return;
    
    // Prevent default behavior for arrow keys to avoid page scrolling
    if (["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"].includes(e.key)) {
      e.preventDefault();
    }

    keysPressed.current.add(e.key);
  }, [isDriving, countdown]);

  const handleKeyUp = useCallback((e: KeyboardEvent) => {
    keysPressed.current.delete(e.key);
  }, []);

  useEffect(() => {
    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("keyup", handleKeyUp);
    
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("keyup", handleKeyUp);
    };
  }, [handleKeyDown, handleKeyUp]);

  // Clear keys when game ends or focus is lost
  useEffect(() => {
    const clearKeys = () => {
      keysPressed.current.clear();
    };

    window.addEventListener("blur", clearKeys);
    window.addEventListener("focus", clearKeys);

    if (!isDriving) {
      clearKeys();
    }

    return () => {
      window.removeEventListener("blur", clearKeys);
      window.removeEventListener("focus", clearKeys);
    };
  }, [isDriving]);

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

  // Mobile button handlers for touch support
  const handleMobileAction = useCallback((action: number[]) => {
    sendAction(action);
  }, [sendAction]);

  return (
    <div
      style={{
        minHeight: "100vh",
        backgroundColor: "#000",
        color: "#FFD700", // Gold
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        flexDirection: "column",
        padding: "2rem",
        fontFamily: "'Segoe UI', sans-serif",
      }}
    >
      {!isDriving && !gameOver && countdown === null && (
        <>
          <h1 style={{ color: "#FFD700", fontSize: "2rem" }}>Enter Your Name</h1>
          <div style={{ display: "flex", alignItems: "center", gap: "1rem", marginTop: "1rem" }}>
            <input
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Driver name"
              style={{
                padding: "0.5rem 1rem",
                backgroundColor: "#111",
                border: "2px solid #FFD700",
                color: "#FFD700",
                borderRadius: "0.5rem",
                fontSize: "1rem",
              }}
            />
            <button
              onClick={start}
              style={{
                padding: "0.5rem 1.2rem",
                backgroundColor: "#FFD700",
                color: "#000",
                fontWeight: "bold",
                borderRadius: "0.5rem",
                border: "none",
                cursor: "pointer",
              }}
            >
              Start Driving
            </button>
          </div>

          <div style={{ 
            marginTop: "1.5rem", 
            textAlign: "center", 
            color: "#AAA",
            fontSize: "0.9rem"
          }}>
            <p><strong>Controls:</strong></p>
            <p>Arrow Keys or WASD to drive</p>
            <p>Hold multiple keys for combined actions</p>
          </div>

          {topScores.length > 0 && (
            <div
              style={{
                marginTop: "2rem",
                backgroundColor: "#111",
                padding: "1rem 1.5rem",
                borderRadius: "1rem",
                maxWidth: "300px",
                boxShadow: "0 0 10px rgba(255, 215, 0, 0.5)",
                color: "#FFD700",
              }}
            >
              <h4 style={{ marginTop: 0 }}>🏆 Top Scores</h4>
              <ol style={{ paddingLeft: "1.2rem", margin: 0 }}>
                {topScores.map((s, i) => {
                  const prefix =
                    i === 0
                      ? "🥇 "
                      : i === 1
                      ? "🥈 "
                      : i === 2
                      ? "🥉 "
                      : `${i + 1}. `;
                  return (
                    <li key={s.session_id} style={{ marginBottom: "0.25rem" }}>
                      {prefix}
                      {s.player || "Anonymous"} – {s.score.toFixed(1)}
                    </li>
                  );
                })}
              </ol>
            </div>
          )}
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
            userSelect: "none",
            WebkitUserSelect: "none",
          }}
        >
          <div>
            <button
              onMouseDown={(e) => {
                e.preventDefault();
                handleMobileAction([0, 1, 0]);
              }}
              onMouseUp={(e) => {
                e.preventDefault();
                handleMobileAction([0, 0, 0]);
              }}
              onMouseLeave={() => handleMobileAction([0, 0, 0])}
              onTouchStart={(e) => {
                e.preventDefault();
                handleMobileAction([0, 1, 0]);
              }}
              onTouchEnd={(e) => {
                e.preventDefault();
                handleMobileAction([0, 0, 0]);
              }}
              onTouchCancel={(e) => {
                e.preventDefault();
                handleMobileAction([0, 0, 0]);
              }}
              onContextMenu={(e) => e.preventDefault()}
              style={{
                ...mobileBtnStyle,
                fontSize: "2rem",
                minWidth: "60px",
                minHeight: "60px",
                border: "2px solid #333",
                borderRadius: "12px",
                backgroundColor: "#f0f0f0",
                cursor: "pointer",
                userSelect: "none",
                WebkitTouchCallout: "none",
                WebkitUserSelect: "none",
                touchAction: "manipulation",
              }}
            >
              ⬆️
            </button>
          </div>
          <div style={{ display: "flex", gap: "1rem" }}>
            <button
              onMouseDown={(e) => {
                e.preventDefault();
                handleMobileAction([-1, 0, 0]);
              }}
              onMouseUp={(e) => {
                e.preventDefault();
                handleMobileAction([0, 0, 0]);
              }}
              onMouseLeave={() => handleMobileAction([0, 0, 0])}
              onTouchStart={(e) => {
                e.preventDefault();
                handleMobileAction([-1, 0, 0]);
              }}
              onTouchEnd={(e) => {
                e.preventDefault();
                handleMobileAction([0, 0, 0]);
              }}
              onTouchCancel={(e) => {
                e.preventDefault();
                handleMobileAction([0, 0, 0]);
              }}
              onContextMenu={(e) => e.preventDefault()}
              style={{
                ...mobileBtnStyle,
                fontSize: "2rem",
                minWidth: "60px",
                minHeight: "60px",
                border: "2px solid #333",
                borderRadius: "12px",
                backgroundColor: "#f0f0f0",
                cursor: "pointer",
                userSelect: "none",
                WebkitTouchCallout: "none",
                WebkitUserSelect: "none",
                touchAction: "manipulation",
              }}
            >
              ⬅️
            </button>
            <button
              onMouseDown={(e) => {
                e.preventDefault();
                handleMobileAction([0, 0, 0.8]);
              }}
              onMouseUp={(e) => {
                e.preventDefault();
                handleMobileAction([0, 0, 0]);
              }}
              onMouseLeave={() => handleMobileAction([0, 0, 0])}
              onTouchStart={(e) => {
                e.preventDefault();
                handleMobileAction([0, 0, 0.8]);
              }}
              onTouchEnd={(e) => {
                e.preventDefault();
                handleMobileAction([0, 0, 0]);
              }}
              onTouchCancel={(e) => {
                e.preventDefault();
                handleMobileAction([0, 0, 0]);
              }}
              onContextMenu={(e) => e.preventDefault()}
              style={{
                ...mobileBtnStyle,
                fontSize: "2rem",
                minWidth: "60px",
                minHeight: "60px",
                border: "2px solid #333",
                borderRadius: "12px",
                backgroundColor: "#ffcccc",
                cursor: "pointer",
                userSelect: "none",
                WebkitTouchCallout: "none",
                WebkitUserSelect: "none",
                touchAction: "manipulation",
              }}
            >
              ⏹️
            </button>
            <button
              onMouseDown={(e) => {
                e.preventDefault();
                handleMobileAction([1, 0, 0]);
              }}
              onMouseUp={(e) => {
                e.preventDefault();
                handleMobileAction([0, 0, 0]);
              }}
              onMouseLeave={() => handleMobileAction([0, 0, 0])}
              onTouchStart={(e) => {
                e.preventDefault();
                handleMobileAction([1, 0, 0]);
              }}
              onTouchEnd={(e) => {
                e.preventDefault();
                handleMobileAction([0, 0, 0]);
              }}
              onTouchCancel={(e) => {
                e.preventDefault();
                handleMobileAction([0, 0, 0]);
              }}
              onContextMenu={(e) => e.preventDefault()}
              style={{
                ...mobileBtnStyle,
                fontSize: "2rem",
                minWidth: "60px",
                minHeight: "60px",
                border: "2px solid #333",
                borderRadius: "12px",
                backgroundColor: "#f0f0f0",
                cursor: "pointer",
                userSelect: "none",
                WebkitTouchCallout: "none",
                WebkitUserSelect: "none",
                touchAction: "manipulation",
              }}
            >
              ➡️
            </button>
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