import { useEffect, useRef, useState } from "react";
import axios from "axios";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL;

console.log("API:", API_BASE_URL);

type PredictionResponse = {
  predictions: number[];
  confidences: number[];
};

export default function App() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [predictions, setPredictions] = useState<number[]>([]);
  const [confidences, setConfidences] = useState<number[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    initializeCanvas();
  }, []);

  const initializeCanvas = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  };

  const getMousePosition = (event: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };

    const rect = canvas.getBoundingClientRect();
    return {
      x: event.clientX - rect.left,
      y: event.clientY - rect.top,
    };
  };

  const startDrawing = (event: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const { x, y } = getMousePosition(event);
    ctx.beginPath();
    ctx.moveTo(x, y);
    setIsDrawing(true);
  };

  const draw = (event: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDrawing) return;

    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const { x, y } = getMousePosition(event);

    ctx.lineWidth = 6;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.strokeStyle = "white";
    ctx.lineTo(x, y);
    ctx.stroke();
  };

  const stopDrawing = () => {
    setIsDrawing(false);
  };

  const clearCanvas = () => {
    initializeCanvas();
    setPredictions([]);
    setConfidences([]);
  };

  const predictDigit = async () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    setLoading(true);

    canvas.toBlob(async (blob) => {
      if (!blob) {
        setLoading(false);
        return;
      }

      const formData = new FormData();
      formData.append("file", blob, "digit.png");

      try {
        const response = await axios.post<PredictionResponse>(
          `${API_BASE_URL}/predict`,
          formData
        );

        setPredictions(response.data.predictions);
        setConfidences(response.data.confidences);
      } catch (error) {
        if (axios.isAxiosError(error)) {
          console.error("Prediction error:", error.message);
          console.error("Status:", error.response?.status);
          console.error("Response data:", error.response?.data);
        } else {
          console.error("Unexpected error:", error);
        }
      } finally {
        setLoading(false);
      }
    }, "image/png");
  };

  const bestPrediction = predictions[0];
  const bestConfidence = confidences[0];

  return (
    <div style={styles.page}>
      <div style={styles.card}>
        <h1 style={styles.title}>MNIST Digit Classifier</h1>
        <p style={styles.subtitle}>Draw a digit and get the model prediction.</p>

        <canvas
          ref={canvasRef}
          width={280}
          height={280}
          style={styles.canvas}
          onMouseDown={startDrawing}
          onMouseMove={draw}
          onMouseUp={stopDrawing}
          onMouseLeave={stopDrawing}
        />

        <div style={styles.buttonRow}>
          <button style={styles.secondaryButton} onClick={clearCanvas}>
            Clear
          </button>
          <button style={styles.primaryButton} onClick={predictDigit} disabled={loading}>
            {loading ? "Predicting..." : "Predict"}
          </button>
        </div>

        {predictions.length > 0 ? (
          <>
            <div style={styles.mainPredictionBox}>
              <p style={styles.mainPredictionLabel}>Best prediction</p>
              <div style={styles.mainPredictionDigit}>{bestPrediction}</div>
              <p style={styles.mainPredictionConfidence}>
                Confidence: {bestConfidence?.toFixed(4)}
              </p>
            </div>

            <div style={styles.resultBox}>
              <p style={styles.resultTitle}>Top 3 predictions</p>

              {predictions.map((pred, index) => (
                <div key={index} style={styles.resultRow}>
                  <span style={styles.rankBadge}>{index + 1}</span>
                  <span style={styles.resultDigit}>Digit {pred}</span>
                  <span style={styles.resultConfidence}>
                    {confidences[index]?.toFixed(4)}
                  </span>
                </div>
              ))}
            </div>
          </>
        ) : (
          <div style={styles.emptyBox}>
            <p style={styles.resultPlaceholder}>No prediction yet.</p>
          </div>
        )}
      </div>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  page: {
    minHeight: "100vh",
    margin: 0,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    background: "#0f172a",
    color: "white",
    fontFamily: "Arial, sans-serif",
    padding: "24px",
  },
  card: {
    background: "#111827",
    padding: "32px",
    borderRadius: "20px",
    boxShadow: "0 12px 30px rgba(0,0,0,0.35)",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    gap: "16px",
    maxWidth: "440px",
    width: "100%",
  },
  title: {
    margin: 0,
    fontSize: "2rem",
    textAlign: "center",
  },
  subtitle: {
    margin: 0,
    color: "#cbd5e1",
    textAlign: "center",
  },
  canvas: {
    border: "3px solid #334155",
    borderRadius: "12px",
    background: "black",
    cursor: "crosshair",
  },
  buttonRow: {
    display: "flex",
    gap: "12px",
    width: "100%",
    justifyContent: "center",
  },
  primaryButton: {
    padding: "12px 20px",
    borderRadius: "10px",
    border: "none",
    background: "#2563eb",
    color: "white",
    fontWeight: 700,
    cursor: "pointer",
  },
  secondaryButton: {
    padding: "12px 20px",
    borderRadius: "10px",
    border: "none",
    background: "#374151",
    color: "white",
    fontWeight: 700,
    cursor: "pointer",
  },
  mainPredictionBox: {
    width: "100%",
    background: "linear-gradient(135deg, #1d4ed8, #2563eb)",
    borderRadius: "18px",
    padding: "20px",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    boxShadow: "0 10px 24px rgba(37, 99, 235, 0.25)",
  },
  mainPredictionLabel: {
    margin: 0,
    fontSize: "0.95rem",
    opacity: 0.9,
  },
  mainPredictionDigit: {
    fontSize: "4.5rem",
    fontWeight: 800,
    lineHeight: 1,
    margin: "8px 0",
  },
  mainPredictionConfidence: {
    margin: 0,
    fontSize: "1rem",
    fontWeight: 600,
  },
  resultBox: {
    width: "100%",
    borderRadius: "14px",
    background: "#1f2937",
    padding: "16px",
    display: "flex",
    flexDirection: "column",
    gap: "10px",
  },
  resultTitle: {
    margin: 0,
    fontWeight: 700,
    fontSize: "1rem",
    textAlign: "center",
  },
  resultRow: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    background: "#111827",
    borderRadius: "10px",
    padding: "10px 12px",
  },
  rankBadge: {
    minWidth: "28px",
    height: "28px",
    borderRadius: "999px",
    background: "#334155",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    fontWeight: 700,
    fontSize: "0.9rem",
  },
  resultDigit: {
    fontWeight: 600,
    flex: 1,
    marginLeft: "12px",
  },
  resultConfidence: {
    color: "#93c5fd",
    fontWeight: 700,
  },
  emptyBox: {
    width: "100%",
    minHeight: "90px",
    borderRadius: "14px",
    background: "#1f2937",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
  },
  resultPlaceholder: {
    margin: 0,
    color: "#94a3b8",
  },
};
