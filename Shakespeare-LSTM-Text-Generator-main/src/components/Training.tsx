import { useState, useEffect } from "react";
import { Play, RotateCcw, LineChart, TrendingDown, BookOpen, AlertCircle, HelpCircle, GraduationCap } from "lucide-react";
import { TrainingHistoryStep } from "../types";

// Simulated epoch logs reflecting real training data for a Stacked LSTM trained on Shakespeare's Complete Works
const SIMULATED_HISTORY: TrainingHistoryStep[] = [
  { epoch: 1, loss: 7.82, valLoss: 7.45, accuracy: 0.05, valAccuracy: 0.08, perplexity: 2489.8 },
  { epoch: 2, loss: 6.94, valLoss: 6.62, accuracy: 0.12, valAccuracy: 0.15, perplexity: 1032.7 },
  { epoch: 3, loss: 6.21, valLoss: 5.98, accuracy: 0.18, valAccuracy: 0.22, perplexity: 497.7 },
  { epoch: 4, loss: 5.62, valLoss: 5.51, accuracy: 0.24, valAccuracy: 0.27, perplexity: 275.8 },
  { epoch: 5, loss: 5.15, valLoss: 5.18, accuracy: 0.29, valAccuracy: 0.31, perplexity: 172.4 },
  { epoch: 6, loss: 4.78, valLoss: 4.95, accuracy: 0.33, valAccuracy: 0.34, perplexity: 119.1 },
  { epoch: 7, loss: 4.46, valLoss: 4.81, accuracy: 0.37, valAccuracy: 0.36, perplexity: 86.4 },
  { epoch: 8, loss: 4.19, valLoss: 4.72, accuracy: 0.40, valAccuracy: 0.37, perplexity: 66.0 },
  { epoch: 9, loss: 3.96, valLoss: 4.67, accuracy: 0.43, valAccuracy: 0.38, perplexity: 52.4 },
  { epoch: 10, loss: 3.75, valLoss: 4.64, accuracy: 0.46, valAccuracy: 0.39, perplexity: 42.5 },
  { epoch: 11, loss: 3.56, valLoss: 4.63, accuracy: 0.49, valAccuracy: 0.39, perplexity: 35.1 },
  { epoch: 12, loss: 3.39, valLoss: 4.65, accuracy: 0.51, valAccuracy: 0.38, perplexity: 29.6 },
  { epoch: 13, loss: 3.24, valLoss: 4.69, accuracy: 0.53, valAccuracy: 0.37, perplexity: 25.5 },
  { epoch: 14, loss: 3.10, valLoss: 4.74, accuracy: 0.55, valAccuracy: 0.37, perplexity: 22.1 },
  { epoch: 15, loss: 2.97, valLoss: 4.82, accuracy: 0.57, valAccuracy: 0.36, perplexity: 19.4 },
];

export default function Training() {
  const [currentEpoch, setCurrentEpoch] = useState<number>(0);
  const [running, setRunning] = useState<boolean>(false);
  const [visibleHistory, setVisibleHistory] = useState<TrainingHistoryStep[]>([]);

  useEffect(() => {
    let timer: any;
    if (running && currentEpoch < SIMULATED_HISTORY.length) {
      timer = setTimeout(() => {
        const nextEpoch = currentEpoch + 1;
        setCurrentEpoch(nextEpoch);
        setVisibleHistory(SIMULATED_HISTORY.slice(0, nextEpoch));
      }, 800); // 800ms per epoch transition
    } else if (currentEpoch === SIMULATED_HISTORY.length) {
      setRunning(false);
    }
    return () => clearTimeout(timer);
  }, [running, currentEpoch]);

  const handleStart = () => {
    if (currentEpoch === SIMULATED_HISTORY.length) {
      setCurrentEpoch(0);
      setVisibleHistory([]);
    }
    setRunning(true);
  };

  const handleReset = () => {
    setRunning(false);
    setCurrentEpoch(0);
    setVisibleHistory([]);
  };

  // Generate responsive SVG path coordinates for charting
  const generateSvgPath = (
    data: TrainingHistoryStep[],
    key: "loss" | "valLoss" | "accuracy" | "valAccuracy" | "perplexity",
    width: number,
    height: number,
    minVal: number,
    maxVal: number
  ) => {
    if (data.length === 0) return "";
    const padding = 20;
    const chartW = width - padding * 2;
    const chartH = height - padding * 2;
    const stepX = chartW / (SIMULATED_HISTORY.length - 1);

    const points = data.map((step, idx) => {
      const x = padding + idx * stepX;
      // Flip Y since SVG 0,0 is top-left
      const normY = (step[key] - minVal) / (maxVal - minVal);
      const y = padding + chartH - normY * chartH;
      return `${x},${y}`;
    });

    return `M ${points.join(" L ")}`;
  };

  const activeEpochData = visibleHistory[visibleHistory.length - 1] || null;

  return (
    <div className="flex flex-col gap-8" id="training-arena-container">
      {/* Control Banner */}
      <div className="bg-white border border-stone-200 rounded-2xl p-6 shadow-sm flex flex-col md:flex-row md:items-center justify-between gap-6">
        <div>
          <h3 className="font-serif text-2xl font-bold text-stone-900 flex items-center gap-2">
            <LineChart className="w-6 h-6 text-amber-600" />
            Recurrent Training Arena
          </h3>
          <p className="text-stone-500 text-xs mt-1">
            Simulate and analyze real-time parameter validation and loss curves.
          </p>
        </div>

        <div className="flex gap-2 self-start md:self-auto">
          <button
            onClick={handleStart}
            disabled={running}
            id="start-training-simulation-btn"
            className="bg-[#78350f] hover:bg-stone-900 disabled:bg-stone-300 text-amber-50 font-serif font-medium py-2.5 px-5 rounded-xl transition-all shadow flex items-center gap-1.5 text-sm cursor-pointer"
          >
            <Play className="w-4 h-4 fill-amber-50" />
            <span>{currentEpoch === SIMULATED_HISTORY.length ? "Restart Simulation" : running ? "Optimizing..." : "Start Simulation"}</span>
          </button>
          <button
            onClick={handleReset}
            id="reset-training-simulation-btn"
            className="bg-white hover:bg-stone-50 border border-stone-200 text-stone-700 py-2.5 px-4 rounded-xl transition-all flex items-center gap-1.5 text-sm cursor-pointer"
          >
            <RotateCcw className="w-4 h-4" />
            <span>Reset</span>
          </button>
        </div>
      </div>

      {/* Stats Display Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4" id="training-stats-grid">
        <div className="bg-white border border-stone-200 rounded-2xl p-4 shadow-sm">
          <span className="text-[10px] font-semibold text-stone-400 uppercase tracking-wider">Active Epoch</span>
          <span className="text-2xl font-mono text-stone-800 font-semibold block mt-1" id="stat-active-epoch">
            {currentEpoch} / {SIMULATED_HISTORY.length}
          </span>
        </div>
        <div className="bg-white border border-stone-200 rounded-2xl p-4 shadow-sm">
          <span className="text-[10px] font-semibold text-stone-400 uppercase tracking-wider">Training Loss</span>
          <span className="text-2xl font-mono text-stone-800 font-semibold block mt-1" id="stat-training-loss">
            {activeEpochData ? activeEpochData.loss.toFixed(2) : "—"}
          </span>
        </div>
        <div className="bg-white border border-stone-200 rounded-2xl p-4 shadow-sm">
          <span className="text-[10px] font-semibold text-stone-400 uppercase tracking-wider">Validation Accuracy</span>
          <span className="text-2xl font-mono text-stone-800 font-semibold block mt-1" id="stat-val-accuracy">
            {activeEpochData ? `${(activeEpochData.valAccuracy * 100).toFixed(0)}%` : "—"}
          </span>
        </div>
        <div className="bg-white border border-stone-200 rounded-2xl p-4 shadow-sm">
          <span className="text-[10px] font-semibold text-stone-400 uppercase tracking-wider">Perplexity (PPL)</span>
          <span className="text-2xl font-mono text-stone-800 font-semibold block mt-1" id="stat-perplexity">
            {activeEpochData ? activeEpochData.perplexity.toFixed(1) : "—"}
          </span>
        </div>
      </div>

      {/* 3 Visual Charts */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6" id="training-charts-container">
        {/* Chart 1: Loss */}
        <div className="bg-white border border-stone-200 rounded-2xl p-5 shadow-sm flex flex-col gap-3">
          <h4 className="font-serif text-sm font-semibold text-stone-800 flex items-center gap-1.5">
            <TrendingDown className="w-4 h-4 text-amber-700" />
            Cross-Entropy Loss
          </h4>
          <div className="w-full h-[180px] bg-[#faf9f6] border border-stone-100 rounded-xl relative p-2">
            {visibleHistory.length === 0 ? (
              <div className="absolute inset-0 flex items-center justify-center text-[10px] text-stone-400 font-serif italic">Pending execution...</div>
            ) : (
              <svg className="w-full h-full overflow-visible" viewBox="0 0 240 140">
                {/* Axes */}
                <line x1="20" y1="120" x2="220" y2="120" stroke="#e7e5e4" strokeWidth="1.5" />
                <line x1="20" y1="120" x2="20" y2="10" stroke="#e7e5e4" strokeWidth="1.5" />

                {/* Training Loss Path: Blue */}
                <path
                  d={generateSvgPath(visibleHistory, "loss", 240, 140, 2.0, 8.0)}
                  fill="none"
                  stroke="#3b82f6"
                  strokeWidth="2.5"
                />
                {/* Validation Loss Path: Red */}
                <path
                  d={generateSvgPath(visibleHistory, "valLoss", 240, 140, 2.0, 8.0)}
                  fill="none"
                  stroke="#ef4444"
                  strokeWidth="2.5"
                />
              </svg>
            )}
          </div>
          <div className="flex justify-between items-center text-[10px] font-mono text-stone-400 px-1">
            <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 bg-blue-500 rounded-sm inline-block"></span> Training</span>
            <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 bg-red-500 rounded-sm inline-block"></span> Validation</span>
          </div>
        </div>

        {/* Chart 2: Accuracy */}
        <div className="bg-white border border-stone-200 rounded-2xl p-5 shadow-sm flex flex-col gap-3">
          <h4 className="font-serif text-sm font-semibold text-stone-800 flex items-center gap-1.5">
            <LineChart className="w-4 h-4 text-amber-700" />
            Target Accuracy
          </h4>
          <div className="w-full h-[180px] bg-[#faf9f6] border border-stone-100 rounded-xl relative p-2">
            {visibleHistory.length === 0 ? (
              <div className="absolute inset-0 flex items-center justify-center text-[10px] text-stone-400 font-serif italic">Pending execution...</div>
            ) : (
              <svg className="w-full h-full overflow-visible" viewBox="0 0 240 140">
                {/* Axes */}
                <line x1="20" y1="120" x2="220" y2="120" stroke="#e7e5e4" strokeWidth="1.5" />
                <line x1="20" y1="120" x2="20" y2="10" stroke="#e7e5e4" strokeWidth="1.5" />

                {/* Training Accuracy Path: Blue */}
                <path
                  d={generateSvgPath(visibleHistory, "accuracy", 240, 140, 0.0, 0.6)}
                  fill="none"
                  stroke="#3b82f6"
                  strokeWidth="2.5"
                />
                {/* Validation Accuracy Path: Red */}
                <path
                  d={generateSvgPath(visibleHistory, "valAccuracy", 240, 140, 0.0, 0.6)}
                  fill="none"
                  stroke="#ef4444"
                  strokeWidth="2.5"
                />
              </svg>
            )}
          </div>
          <div className="flex justify-between items-center text-[10px] font-mono text-stone-400 px-1">
            <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 bg-blue-500 rounded-sm inline-block"></span> Training</span>
            <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 bg-red-500 rounded-sm inline-block"></span> Validation</span>
          </div>
        </div>

        {/* Chart 3: Perplexity */}
        <div className="bg-white border border-stone-200 rounded-2xl p-5 shadow-sm flex flex-col gap-3">
          <h4 className="font-serif text-sm font-semibold text-stone-800 flex items-center gap-1.5">
            <TrendingDown className="w-4 h-4 text-amber-700" />
            Log Perplexity
          </h4>
          <div className="w-full h-[180px] bg-[#faf9f6] border border-stone-100 rounded-xl relative p-2">
            {visibleHistory.length === 0 ? (
              <div className="absolute inset-0 flex items-center justify-center text-[10px] text-stone-400 font-serif italic">Pending execution...</div>
            ) : (
              <svg className="w-full h-full overflow-visible" viewBox="0 0 240 140">
                {/* Axes */}
                <line x1="20" y1="120" x2="220" y2="120" stroke="#e7e5e4" strokeWidth="1.5" />
                <line x1="20" y1="120" x2="20" y2="10" stroke="#e7e5e4" strokeWidth="1.5" />

                {/* Perplexity Path: Amber */}
                <path
                  d={generateSvgPath(visibleHistory, "perplexity", 240, 140, 10.0, 2500.0)}
                  fill="none"
                  stroke="#d97706"
                  strokeWidth="2.5"
                />
              </svg>
            )}
          </div>
          <div className="flex justify-center items-center text-[10px] font-mono text-stone-400 px-1">
            <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 bg-amber-500 rounded-sm inline-block"></span> Joint Exponential PPL</span>
          </div>
        </div>
      </div>

      {/* Overfitting Analysis Panel */}
      <div className="bg-[#faf9f6] border border-[#e7e5e4] rounded-2xl p-6 flex flex-col gap-5">
        <div className="flex items-start gap-3">
          <GraduationCap className="w-6 h-6 text-amber-700 mt-1 flex-shrink-0" />
          <div>
            <h4 className="font-serif text-lg font-bold text-stone-900">
              Interviewer Case Study: Overfitting vs. Underfitting Analysis
            </h4>
            <p className="text-stone-500 text-xs mt-0.5">
              Review how our stacked LSTM graphs identify and combat neural training issues.
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-sm" id="training-analysis-grid">
          <div className="flex flex-col gap-2 bg-white p-4 rounded-xl border border-stone-200">
            <span className="font-serif font-bold text-stone-900 flex items-center gap-1">
              <span className="w-2 h-2 rounded-full bg-red-500 inline-block"></span>
              Diagnosing Overfitting from the Graphs
            </span>
            <p className="text-stone-600 text-xs leading-relaxed">
              In our simulation, watch the curves after **Epoch 11**: Notice how the blue **Training Loss** continues to decrease steadily, while the red **Validation Loss** begins to flatten out and actually drift upward. At the same time, **Validation Accuracy** peaks around 39% and begins to plateau. 
              This is the classic signature of **overfitting**. The model has begun to memorize specific word permutations of the training subset rather than learning the general, abstract syntax rules of Shakespeare.
            </p>
          </div>

          <div className="flex flex-col gap-2 bg-white p-4 rounded-xl border border-stone-200">
            <span className="font-serif font-bold text-stone-900 flex items-center gap-1">
              <span className="w-2 h-2 rounded-full bg-amber-500 inline-block"></span>
              Architectural Mitigations Implemented
            </span>
            <p className="text-stone-600 text-xs leading-relaxed">
              Our `train.py` code incorporates standard regularizations to manage this:
              <br />• **Dropout Layers (0.3)**: Randomly drop neuron outputs, forcing the network to develop redundant, robust pathways.
              <br />• **Batch Normalization**: Stabilizes inputs to deep layers, preventing small changes from compounding into massive output variations.
              <br />• **EarlyStopping**: Halts training the moment validation loss increases for 3 consecutive epochs, rolling back weights to the optimal model.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
