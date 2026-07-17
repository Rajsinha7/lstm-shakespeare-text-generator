import { useState } from "react";
import { motion } from "motion/react";
import { Sliders, RefreshCw, Key, HelpCircle, Layers, ShieldCheck } from "lucide-react";

interface GateDetails {
  name: string;
  symbol: string;
  formula: string;
  description: string;
  interviewInsight: string;
  activationRange: string;
}

const GATES: GateDetails[] = [
  {
    name: "Forget Gate",
    symbol: "f_t",
    formula: "f_t = \\sigma(W_f \\cdot [h_{t-1}, x_t] + b_f)",
    description: "Determines what proportion of the previous cell memory (long-term historical state) should be deleted. It outputs a coefficient vector between 0 (completely forget) and 1 (completely preserve) by feeding the current input x_t and preceding hidden state h_{t-1} into a Sigmoid layer.",
    interviewInsight: "Allows the model to drop stale contexts. For instance, if a paragraph shifts from singular to plural or the subject changes, the forget gate discards outdated grammar cues.",
    activationRange: "[0, 1] (Sigmoid Gate Control)"
  },
  {
    name: "Input Gate & Candidate State",
    symbol: "i_t \\odot \\tilde{C}_t",
    formula: "i_t = \\sigma(W_i \\cdot [h_{t-1}, x_t] + b_i) \\quad \\tilde{C}_t = \\tanh(W_c \\cdot [h_{t-1}, x_t] + b_c)",
    description: "Determines which new semantic details should be committed to the cell memory. The sigmoid gate i_t controls the flow magnitude, while the tanh layer \\tilde{C}_t proposes candidate vectors ranging from -1 to 1 to scale the updates.",
    interviewInsight: "Prevents noise from entering cell memory. Sigmoid screens which words are worth saving; tanh prepares the balanced values to add.",
    activationRange: "i_t \\in [0, 1], \\tilde{C}_t \\in [-1, 1]"
  },
  {
    name: "Cell State Update",
    symbol: "C_t",
    formula: "C_t = f_t \\odot C_{t-1} + i_t \\odot \\tilde{C}_t",
    description: "The core conveyor belt of long-term memory. It scales the previous state C_{t-1} by the forget factors, then adds the new screened candidates in a purely linear operations loop. This avoids multiplicative decay down the temporal sequence.",
    interviewInsight: "This linear cell state highway is what resolves the Vanishing Gradient problem. Gradients backpropagate easily because there are no recursive matrix multiplications.",
    activationRange: "Unbounded continuous scalar memory belt"
  },
  {
    name: "Output Gate & Hidden State",
    symbol: "o_t \\to h_t",
    formula: "o_t = \\sigma(W_o \\cdot [h_{t-1}, x_t] + b_o) \\quad h_t = o_t \\odot \\tanh(C_t)",
    description: "Determines the visible Hidden State output h_t passed to the next block or softmax prediction layer. A sigmoid gate o_t screens what features are extracted, which are then multiplied by the cell memory scaled back into [-1, 1] using tanh.",
    interviewInsight: "The cell state holds all potential memory, but only relevant information is output as the active hidden state at the current time step.",
    activationRange: "h_t \\in [-1, 1] (Bounded visible output)"
  }
];

export default function Sandbox() {
  const [activeGateIndex, setActiveGateIndex] = useState(0);
  const activeGate = GATES[activeGateIndex];

  return (
    <div className="bg-[#fafaf9] border border-[#e7e5e4] rounded-2xl p-6 shadow-sm" id="lstm-sandbox-container">
      <div className="flex flex-col md:flex-row md:items-center justify-between border-b border-stone-200 pb-4 mb-6 gap-3">
        <div>
          <h3 className="font-serif text-2xl font-bold text-stone-900 flex items-center gap-2">
            <Layers className="w-6 h-6 text-amber-600" />
            LSTM Cell Gate Explorer
          </h3>
          <p className="text-stone-500 text-xs mt-1">
            Analyze mathematical transformations, vector signals, and gating networks step-by-step.
          </p>
        </div>
        <span className="text-[10px] text-amber-800 bg-amber-50 border border-amber-200 px-3 py-1 rounded-full font-mono font-medium self-start md:self-auto">
          Gate Type: LSTM (Long Short-Term Memory)
        </span>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Gate Selection: 4 cols */}
        <div className="lg:col-span-4 flex flex-col gap-2">
          <label className="text-xs font-semibold text-stone-700 uppercase tracking-wider mb-1">Select Cell Gate Block</label>
          {GATES.map((gate, idx) => (
            <button
              key={gate.name}
              id={`sandbox-gate-btn-${idx}`}
              onClick={() => setActiveGateIndex(idx)}
              className={`text-left p-4 rounded-xl border transition-all flex justify-between items-center cursor-pointer ${
                activeGateIndex === idx
                  ? "bg-stone-900 border-stone-900 text-white shadow-md font-semibold"
                  : "bg-white border-stone-200 text-stone-700 hover:bg-stone-50"
              }`}
            >
              <div className="flex flex-col">
                <span className="text-xs font-mono text-stone-400 font-normal">{gate.symbol}</span>
                <span className="text-sm mt-0.5">{gate.name}</span>
              </div>
              <span className={`text-[10px] font-mono px-1.5 py-0.5 rounded ${activeGateIndex === idx ? "bg-amber-600 text-white" : "bg-stone-100 text-stone-500"}`}>
                G{idx + 1}
              </span>
            </button>
          ))}
        </div>

        {/* Dynamic Gate Display & Visualization: 8 cols */}
        <div className="lg:col-span-8 flex flex-col gap-6">
          {/* Animated SVG Diagram representing active gate */}
          <div className="bg-white border border-stone-200 rounded-xl p-6 shadow-sm flex items-center justify-center relative overflow-hidden h-[180px]" id="cell-gate-diagram-box">
            <div className="absolute top-3 left-3 text-[10px] uppercase tracking-widest text-stone-400 font-mono">Cell Activation Diagram</div>

            <svg className="w-full max-w-md h-full" viewBox="0 0 400 120" fill="none" xmlns="http://www.w3.org/2000/svg">
              {/* Main State horizontal stream */}
              <line x1="20" y1="60" x2="380" y2="60" stroke="#e7e5e4" strokeWidth="4" strokeDasharray="6 6" />
              
              {/* Highlight Cell State highway on G3 */}
              <line
                x1="20"
                y1="60"
                x2="380"
                y2="60"
                stroke={activeGateIndex === 2 ? "#d97706" : "#cbd5e1"}
                strokeWidth={activeGateIndex === 2 ? "4" : "1.5"}
                className={activeGateIndex === 2 ? "transition-all" : ""}
              />

              {/* Node Inputs left */}
              <g id="input-node">
                <circle cx="50" cy="60" r="16" fill="#f5f3ef" stroke="#1c1917" strokeWidth="2" />
                <text x="50" y="64" fill="#1c1917" fontSize="10" fontFamily="monospace" textAnchor="middle">h_t-1</text>
              </g>

              {/* Node Input bottom */}
              <g id="current-input-node">
                <circle cx="160" cy="100" r="14" fill="#f5f3ef" stroke="#1c1917" strokeWidth="2" />
                <text x="160" y="104" fill="#1c1917" fontSize="10" fontFamily="monospace" textAnchor="middle">x_t</text>
              </g>

              {/* Activation Gate Gate G1 */}
              <g id="activation-gate-node">
                <rect
                  x="140"
                  y="45"
                  width="40"
                  height="30"
                  rx="6"
                  fill={activeGateIndex === 0 ? "#78350f" : "#fafaf9"}
                  stroke={activeGateIndex === 0 ? "#78350f" : "#a8a29e"}
                  strokeWidth="2"
                  className="transition-colors duration-300"
                />
                <text x="160" y="63" fill={activeGateIndex === 0 ? "#ffffff" : "#1c1917"} fontSize="12" fontFamily="serif" textAnchor="middle" fontWeight="bold">
                  {activeGateIndex === 0 ? "σ" : "Forget"}
                </text>
              </g>

              {/* Activation Gate Gate G2 */}
              <g id="input-gate-node">
                <rect
                  x="210"
                  y="45"
                  width="40"
                  height="30"
                  rx="6"
                  fill={activeGateIndex === 1 ? "#78350f" : "#fafaf9"}
                  stroke={activeGateIndex === 1 ? "#78350f" : "#a8a29e"}
                  strokeWidth="2"
                  className="transition-colors duration-300"
                />
                <text x="230" y="63" fill={activeGateIndex === 1 ? "#ffffff" : "#1c1917"} fontSize="11" fontFamily="serif" textAnchor="middle" fontWeight="bold">
                  {activeGateIndex === 1 ? "σ × tanh" : "Input"}
                </text>
              </g>

              {/* Update Cell Node G3 */}
              <g id="cell-state-update-node">
                <circle
                  cx="290"
                  y="60"
                  r="16"
                  fill={activeGateIndex === 2 ? "#d97706" : "#fafaf9"}
                  stroke={activeGateIndex === 2 ? "#d97706" : "#a8a29e"}
                  strokeWidth="2"
                  className="transition-colors duration-300"
                />
                <text x="290" y="64" fill={activeGateIndex === 2 ? "#ffffff" : "#1c1917"} fontSize="11" fontFamily="monospace" textAnchor="middle">
                  C_t
                </text>
              </g>

              {/* Extraction Output Node G4 */}
              <g id="output-gate-node">
                <rect
                  x="330"
                  y="45"
                  width="40"
                  height="30"
                  rx="6"
                  fill={activeGateIndex === 3 ? "#78350f" : "#fafaf9"}
                  stroke={activeGateIndex === 3 ? "#78350f" : "#a8a29e"}
                  strokeWidth="2"
                  className="transition-colors duration-300"
                />
                <text x="350" y="63" fill={activeGateIndex === 3 ? "#ffffff" : "#1c1917"} fontSize="11" fontFamily="serif" textAnchor="middle" fontWeight="bold">
                  {activeGateIndex === 3 ? "σ × tanh" : "Output"}
                </text>
              </g>

              {/* Connecting lines */}
              <path d="M 160 86 L 160 75" stroke="#1c1917" strokeWidth="1.5" markerEnd="url(#arrow)" />
              <path d="M 160 86 L 230 86 L 230 75" stroke="#1c1917" strokeWidth="1.5" fill="none" />
              
              {/* Dynamic Animated Vector dots */}
              <motion.circle
                cx="50"
                cy="60"
                r="4"
                fill="#d97706"
                animate={{ cx: [50, 140, 210, 290, 330, 380] }}
                transition={{ duration: 4, repeat: Infinity, ease: "linear" }}
              />
            </svg>
          </div>

          {/* Gate Formula, description, and Interview Tips */}
          <div className="bg-[#faf9f6] border border-[#e7e5e4] rounded-xl p-5 flex flex-col gap-4">
            {/* Symbol & Formula banner */}
            <div className="flex justify-between items-center border-b border-stone-200 pb-3 flex-wrap gap-2">
              <span className="font-serif font-bold text-stone-900 text-base">{activeGate.name}</span>
              <div className="bg-stone-900 text-stone-100 font-mono text-xs px-3 py-1 rounded-md">
                {activeGate.formula}
              </div>
            </div>

            {/* Description */}
            <div className="text-stone-700 text-sm leading-relaxed" id="sandbox-gate-description">
              {activeGate.description}
            </div>

            {/* Details Table */}
            <div className="grid grid-cols-2 gap-4 border-y border-stone-200 py-3 text-xs">
              <div>
                <span className="text-stone-400 block uppercase tracking-wider font-semibold">Value Active Range</span>
                <span className="font-mono text-stone-800 font-medium mt-0.5 block">{activeGate.activationRange}</span>
              </div>
              <div>
                <span className="text-stone-400 block uppercase tracking-wider font-semibold">Signal Character</span>
                <span className="font-mono text-stone-800 font-medium mt-0.5 block">Dynamic non-linear activation</span>
              </div>
            </div>

            {/* Interview Insights */}
            <div className="bg-amber-50 border border-amber-200 p-4 rounded-lg text-xs" id="sandbox-gate-interview-insight">
              <span className="font-bold text-amber-900 uppercase tracking-wide flex items-center gap-1 mb-1">
                <ShieldCheck className="w-4 h-4 text-amber-700" />
                Internship Interviewer Insight
              </span>
              <p className="text-amber-800 leading-relaxed font-sans">{activeGate.interviewInsight}</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
