import { useState } from "react";
import { motion, AnimatePresence } from "motion/react";
import { Sparkles, HelpCircle, AlertTriangle, ArrowRight, BookOpen, Layers, Check, Loader2 } from "lucide-react";
import { GenerationResponse, TokenDistribution } from "../types";

const PRESET_SEEDS = [
  { label: "Sonnet 18", text: "Shall I compare thee to a summer's day" },
  { label: "Hamlet", text: "To be, or not to be, that is the question" },
  { label: "Romeo & Juliet", text: "O Romeo Romeo wherefore art thou Romeo" },
  { label: "As You Like It", text: "All the world's a stage and all the men" },
  { label: "Macbeth", text: "Double double toil and trouble fire burn and cauldron" },
];

export default function Playground() {
  const [seed, setSeed] = useState(PRESET_SEEDS[0].text);
  const [modelType, setModelType] = useState<"Single LSTM" | "Stacked LSTM">("Stacked LSTM");
  const [method, setMethod] = useState<"Temperature Sampling" | "Beam Search">("Temperature Sampling");
  const [length, setLength] = useState(25);
  const [temperature, setTemperature] = useState(0.5);
  const [beamWidth, setBeamWidth] = useState(3);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<GenerationResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedWordIndex, setSelectedWordIndex] = useState<number | null>(null);

  const handleGenerate = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    setSelectedWordIndex(null);

    try {
      const response = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          seedText: seed,
          length,
          temperature,
          modelType,
          method,
          beamWidth,
        }),
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.error || "Generation failed");
      }

      const data: GenerationResponse = await response.json();
      setResult(data);
      // Select the first word by default to show probability distributions
      if (data.tokenDistributions && data.tokenDistributions.length > 0) {
        setSelectedWordIndex(0);
      }
    } catch (err: any) {
      setError(err.message || "An unexpected error occurred.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-8" id="playground-container">
      {/* Parameters Form: 5 cols */}
      <div className="lg:col-span-5 bg-[#f5f3ef] border border-[#e7e5e4] rounded-2xl p-6 shadow-sm flex flex-col gap-6">
        <div>
          <h3 className="font-serif text-xl font-semibold text-stone-900 flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-amber-600" />
            Generator Settings
          </h3>
          <p className="text-stone-500 text-xs mt-1">Configure parameters for next-word sequence generation.</p>
        </div>

        {/* Preset Seeds */}
        <div className="flex flex-col gap-2">
          <label className="text-xs font-semibold text-stone-700 uppercase tracking-wider">Select Seed Preset</label>
          <div className="flex flex-wrap gap-1.5">
            {PRESET_SEEDS.map((preset) => (
              <button
                key={preset.label}
                id={`preset-btn-${preset.label.toLowerCase().replace(/\s+/g, '-')}`}
                onClick={() => setSeed(preset.text)}
                className={`text-xs px-3 py-1.5 rounded-lg border transition-all ${
                  seed === preset.text
                    ? "bg-amber-100 border-amber-300 text-amber-900 font-medium"
                    : "bg-white border-stone-200 text-stone-600 hover:bg-stone-50"
                }`}
              >
                {preset.label}
              </button>
            ))}
          </div>
        </div>

        {/* Custom Input */}
        <div className="flex flex-col gap-1.5">
          <label className="text-xs font-semibold text-stone-700 uppercase tracking-wider">Custom Seed Sentence</label>
          <textarea
            id="seed-input-textarea"
            value={seed}
            onChange={(e) => setSeed(e.target.value)}
            rows={2}
            className="w-full text-sm p-3 bg-white border border-stone-200 rounded-xl focus:outline-none focus:ring-1 focus:ring-amber-500 font-sans"
            placeholder="Type your Shakespearean seed text here..."
          />
        </div>

        {/* Model Selection */}
        <div className="flex flex-col gap-1.5">
          <label className="text-xs font-semibold text-stone-700 uppercase tracking-wider">LSTM Architecture</label>
          <div className="grid grid-cols-2 gap-2">
            <button
              id="model-type-single-btn"
              onClick={() => setModelType("Single LSTM")}
              className={`p-3 text-xs rounded-xl border text-center transition-all flex flex-col items-center gap-1 ${
                modelType === "Single LSTM"
                  ? "bg-white border-amber-500 text-stone-900 shadow-sm ring-1 ring-amber-500 font-medium"
                  : "bg-white/50 border-stone-200 text-stone-500 hover:bg-white"
              }`}
            >
              <Layers className="w-4 h-4 text-stone-400" />
              <span>Single Layer LSTM</span>
            </button>
            <button
              id="model-type-stacked-btn"
              onClick={() => setModelType("Stacked LSTM")}
              className={`p-3 text-xs rounded-xl border text-center transition-all flex flex-col items-center gap-1 ${
                modelType === "Stacked LSTM"
                  ? "bg-white border-amber-500 text-stone-900 shadow-sm ring-1 ring-amber-500 font-medium"
                  : "bg-white/50 border-stone-200 text-stone-500 hover:bg-white"
              }`}
            >
              <Layers className="w-4 h-4 text-amber-600" />
              <span>Stacked LSTM (2 Layers)</span>
            </button>
          </div>
        </div>

        {/* Decoding Method */}
        <div className="flex flex-col gap-1.5">
          <label className="text-xs font-semibold text-stone-700 uppercase tracking-wider">Decoding Strategy</label>
          <div className="grid grid-cols-2 gap-2">
            <button
              id="decoding-strategy-temp-btn"
              onClick={() => setMethod("Temperature Sampling")}
              className={`p-2.5 text-xs rounded-xl border transition-all ${
                method === "Temperature Sampling"
                  ? "bg-stone-900 border-stone-900 text-white font-medium shadow-sm"
                  : "bg-white border-stone-200 text-stone-600 hover:bg-stone-50"
              }`}
            >
              Temperature Sampling
            </button>
            <button
              id="decoding-strategy-beam-btn"
              onClick={() => setMethod("Beam Search")}
              className={`p-2.5 text-xs rounded-xl border transition-all ${
                method === "Beam Search"
                  ? "bg-stone-900 border-stone-900 text-white font-medium shadow-sm"
                  : "bg-white border-stone-200 text-stone-600 hover:bg-stone-50"
              }`}
            >
              Beam Search
            </button>
          </div>
        </div>

        {/* Dynamic Sliders */}
        <div className="flex flex-col gap-4 bg-white border border-stone-200 rounded-xl p-4">
          <div className="flex flex-col gap-1.5">
            <div className="flex justify-between text-xs font-semibold text-stone-700">
              <span className="uppercase tracking-wider">Words to Predict</span>
              <span className="text-amber-700 font-mono font-medium">{length} words</span>
            </div>
            <input
              id="predict-length-slider"
              type="range"
              min={10}
              max={60}
              step={5}
              value={length}
              onChange={(e) => setLength(Number(e.target.value))}
              className="w-full accent-amber-600 cursor-pointer"
            />
          </div>

          {method === "Temperature Sampling" ? (
            <div className="flex flex-col gap-1.5">
              <div className="flex justify-between text-xs font-semibold text-stone-700">
                <span className="uppercase tracking-wider flex items-center gap-1">
                  Temperature
                  <HelpCircle className="w-3.5 h-3.5 text-stone-400" title="Alters predictions distribution scaling. Low is deterministic, High is creative." />
                </span>
                <span className="text-amber-700 font-mono font-medium">{temperature.toFixed(1)}</span>
              </div>
              <input
                id="temperature-slider"
                type="range"
                min={0.1}
                max={1.5}
                step={0.1}
                value={temperature}
                onChange={(e) => setTemperature(Number(e.target.value))}
                className="w-full accent-amber-600 cursor-pointer"
              />
              <div className="flex justify-between text-[10px] text-stone-400 font-mono">
                <span>0.2 (Safe/Repetitive)</span>
                <span>1.0 (Balanced)</span>
                <span>1.5 (Creative/Chaos)</span>
              </div>
            </div>
          ) : (
            <div className="flex flex-col gap-1.5">
              <div className="flex justify-between text-xs font-semibold text-stone-700">
                <span className="uppercase tracking-wider flex items-center gap-1">
                  Beam Width
                  <HelpCircle className="w-3.5 h-3.5 text-stone-400" title="Defines number of concurrently evaluated continuous token branches." />
                </span>
                <span className="text-amber-700 font-mono font-medium">{beamWidth} active paths</span>
              </div>
              <input
                id="beam-width-slider"
                type="range"
                min={2}
                max={5}
                step={1}
                value={beamWidth}
                onChange={(e) => setBeamWidth(Number(e.target.value))}
                className="w-full accent-amber-600 cursor-pointer"
              />
              <div className="flex justify-between text-[10px] text-stone-400 font-mono">
                <span>Narrow (Speedy)</span>
                <span>Broad (Exhaustive Search)</span>
              </div>
            </div>
          )}
        </div>

        {/* Generate Button */}
        <button
          id="trigger-generate-btn"
          onClick={handleGenerate}
          disabled={loading}
          className="w-full bg-[#78350f] hover:bg-stone-900 text-amber-50 disabled:bg-stone-300 font-serif font-medium py-3.5 px-6 rounded-xl transition-all shadow-md flex items-center justify-center gap-2 text-base cursor-pointer"
        >
          {loading ? (
            <>
              <Loader2 className="w-5 h-5 animate-spin" />
              <span>Transcribing Scroll...</span>
            </>
          ) : (
            <>
              <BookOpen className="w-5 h-5" />
              <span>Generate Shakespearean Verse</span>
            </>
          )}
        </button>
      </div>

      {/* Output Display Panel: 7 cols */}
      <div className="lg:col-span-7 flex flex-col gap-6">
        {/* Results Screen */}
        <div className="bg-white border border-stone-200 rounded-2xl p-6 shadow-sm min-h-[300px] flex flex-col justify-between">
          <div className="flex flex-col gap-4">
            <div className="flex justify-between items-center border-b border-stone-100 pb-3">
              <span className="text-xs font-semibold text-stone-400 uppercase tracking-widest font-mono">Output Verse Scroll</span>
              <span className="text-xs text-stone-500 bg-stone-100 px-2.5 py-1 rounded-full font-mono">
                {modelType} • {method}
              </span>
            </div>

            {/* Error Message */}
            {error && (
              <div id="generation-error-box" className="p-4 bg-red-50 border border-red-200 text-red-800 rounded-xl flex items-start gap-2.5 text-xs">
                <AlertTriangle className="w-4 h-4 text-red-600 flex-shrink-0 mt-0.5" />
                <div>
                  <span className="font-semibold">Simulated Server Exception:</span> {error}
                </div>
              </div>
            )}

            {/* Welcome State */}
            {!loading && !result && !error && (
              <div id="playground-empty-state" className="py-12 text-center text-stone-400 flex flex-col items-center justify-center gap-3">
                <BookOpen className="w-12 h-12 text-stone-200 stroke-[1.5]" />
                <p className="font-serif italic text-stone-500 text-lg">"Be not afraid of greatness: some are born great..."</p>
                <p className="text-stone-400 text-xs max-w-sm">
                  Click the amber button on the left to activate our stacked Recurrent Neural Network and predict the next words.
                </p>
              </div>
            )}

            {/* Loading Placeholder */}
            {loading && (
              <div id="generation-loading-skeleton" className="py-16 flex flex-col items-center gap-3">
                <Loader2 className="w-8 h-8 text-amber-600 animate-spin" />
                <p className="font-serif italic text-stone-600 text-sm">Evaluating hidden cell weights...</p>
                <div className="w-2/3 h-2 bg-stone-100 rounded-full overflow-hidden">
                  <motion.div
                    className="h-full bg-amber-600 rounded-full"
                    initial={{ width: "0%" }}
                    animate={{ width: "100%" }}
                    transition={{ duration: 3, repeat: Infinity }}
                  />
                </div>
              </div>
            )}

            {/* Simulated Prediction Results */}
            {result && (
              <div className="flex flex-col gap-6">
                {/* Generated Verse Text */}
                <div id="output-verse-display" className="p-5 bg-[#faf9f6] border border-[#e7e5e4] rounded-xl font-serif text-lg leading-relaxed text-stone-900 relative shadow-inner overflow-hidden">
                  <div className="absolute right-3 top-3 opacity-5 pointer-events-none select-none font-serif text-8xl">”</div>
                  {/* Words rendering as interactive triggers */}
                  <div className="flex flex-wrap gap-x-1.5 gap-y-1">
                    {/* Render Seed Words First */}
                    <span className="text-stone-500 font-normal italic">{seed}</span>
                    <ArrowRight className="w-4 h-4 self-center text-stone-300 flex-shrink-0" />
                    
                    {/* Predicted Words */}
                    {result.tokenDistributions.map((token, idx) => (
                      <motion.span
                        key={idx}
                        id={`generated-word-${idx}`}
                        initial={{ opacity: 0, y: 3 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: idx * 0.04 }}
                        onClick={() => setSelectedWordIndex(idx)}
                        className={`cursor-pointer px-1.5 py-0.5 rounded transition-all select-none border-b-2 ${
                          selectedWordIndex === idx
                            ? "bg-amber-100 text-amber-900 border-amber-500 font-semibold"
                            : "hover:bg-stone-100 text-stone-900 border-stone-200"
                        }`}
                      >
                        {token.word}
                      </motion.span>
                    ))}
                  </div>
                </div>

                <div className="text-[10px] text-stone-400 flex items-center gap-1 italic">
                  <span>💡 Tip:</span>
                  <span>Click on any underlined word inside the scroll to view its Softmax alternative predictions!</span>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Dynamic Distribution / Probability Visuals */}
        <AnimatePresence mode="wait">
          {result && selectedWordIndex !== null && (
            <motion.div
              key={selectedWordIndex}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="bg-white border border-stone-200 rounded-2xl p-5 shadow-sm"
              id="softmax-distribution-container"
            >
              <div className="flex justify-between items-center border-b border-stone-100 pb-3 mb-4">
                <div>
                  <h4 className="font-serif text-sm font-semibold text-stone-900">
                    Softmax Probability Log (Token #{selectedWordIndex + 1})
                  </h4>
                  <p className="text-stone-400 text-[10px]">
                    Chosen word: <span className="font-mono font-semibold text-stone-800">"{result.tokenDistributions[selectedWordIndex].word}"</span>
                  </p>
                </div>
                <span className="text-[10px] text-stone-500 font-mono bg-stone-100 px-2 py-0.5 rounded">
                  Vocabulary Size: 8,421 words
                </span>
              </div>

              {/* Bars rendering alternative predictions */}
              <div className="flex flex-col gap-3">
                {result.tokenDistributions[selectedWordIndex].alternatives.map((alt, aIdx) => {
                  const isChosen = alt.token === result.tokenDistributions[selectedWordIndex!].word;
                  return (
                    <div key={aIdx} className="flex flex-col gap-1.5" id={`softmax-token-bar-${aIdx}`}>
                      <div className="flex justify-between text-xs font-mono">
                        <span className={`font-semibold ${isChosen ? "text-amber-900" : "text-stone-600"}`}>
                          "{alt.token}" {isChosen && <span className="text-[10px] text-amber-600 font-serif font-normal italic">(Selected)</span>}
                        </span>
                        <span className="text-stone-500">{(alt.probability * 100).toFixed(1)}%</span>
                      </div>
                      <div className="w-full h-2.5 bg-stone-100 rounded-full overflow-hidden relative border border-stone-200">
                        <motion.div
                          initial={{ width: 0 }}
                          animate={{ width: `${alt.probability * 100}%` }}
                          transition={{ duration: 0.6, ease: "easeOut" }}
                          className={`h-full rounded-full ${isChosen ? "bg-amber-600" : "bg-stone-400"}`}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Beam Search Tree expansion (if enabled) */}
        {result && method === "Beam Search" && result.beamSearchSteps && result.beamSearchSteps.length > 0 && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="bg-stone-950 border border-stone-800 text-stone-300 rounded-2xl p-5 shadow-lg"
            id="beam-search-tree-container"
          >
            <div className="flex items-center gap-2 border-b border-stone-800 pb-3 mb-4">
              <span className="p-1 rounded bg-stone-800 text-stone-400 font-mono text-[10px] uppercase font-bold tracking-wider">Log</span>
              <div>
                <h4 className="font-serif text-sm font-semibold text-stone-100">Active Beam Search Branching Tree</h4>
                <p className="text-stone-500 text-[10px]">Step-by-step joint cumulative probability calculation.</p>
              </div>
            </div>

            <div className="flex flex-col gap-4 font-mono text-[11px] max-h-[220px] overflow-y-auto pr-2 custom-scrollbar">
              {result.beamSearchSteps.map((stepData) => (
                <div key={stepData.step} className="border-l-2 border-stone-800 pl-3 flex flex-col gap-1.5" id={`beam-step-${stepData.step}`}>
                  <div className="text-stone-500 font-semibold text-[10px]">STEP {stepData.step} / {length}</div>
                  {stepData.candidates.map((cand, cIdx) => (
                    <div key={cIdx} className="flex justify-between items-center bg-stone-900/50 hover:bg-stone-900 border border-stone-900 px-3 py-1.5 rounded-lg transition-all">
                      <span className="text-stone-300 truncate max-w-[80%]">
                        ... <span className="text-amber-500 font-semibold">{" " + cand.sequence.split(" ").slice(-3).join(" ")}</span>
                      </span>
                      <span className="text-stone-500 font-semibold text-[10px] bg-stone-950 px-2 py-0.5 rounded">
                        Joint Log-Score: {cand.score.toFixed(2)}
                      </span>
                    </div>
                  ))}
                </div>
              ))}
            </div>
          </motion.div>
        )}
      </div>
    </div>
  );
}
