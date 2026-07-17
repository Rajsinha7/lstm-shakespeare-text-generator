import { useState } from "react";
import { motion, AnimatePresence } from "motion/react";
import { Feather, Download, Layers, ShieldCheck, PlayCircle, GraduationCap, Code2, AlertTriangle } from "lucide-react";

// Sub-components import
import Playground from "./components/Playground";
import Sandbox from "./components/Sandbox";
import Explorer from "./components/Explorer";
import Training from "./components/Training";
import Interview from "./components/Interview";

type TabID = "playground" | "sandbox" | "explorer" | "training" | "interview";

export default function App() {
  const [activeTab, setActiveTab] = useState<TabID>("playground");
  const [isZipping, setIsZipping] = useState(false);

  const handleDownloadCodebase = () => {
    setIsZipping(true);
    try {
      window.location.href = "/api/download-zip";
    } catch (err) {
      console.error(err);
    } finally {
      setTimeout(() => setIsZipping(false), 3000);
    }
  };

  return (
    <div className="min-h-screen bg-[#fbfaf7] text-stone-900 font-sans antialiased pb-24 border-t-4 border-[#78350f]">
      {/* Editorial Scholarly Header Masthead */}
      <header className="max-w-7xl mx-auto px-4 pt-10 pb-8 border-b border-stone-200 flex flex-col md:flex-row md:items-end justify-between gap-6" id="app-header-masthead">
        <div className="flex flex-col gap-2">
          <div className="flex items-center gap-2 text-[#78350f]">
            <Feather className="w-6 h-6 stroke-[1.5]" />
            <span className="font-mono text-xs uppercase tracking-widest font-bold">The Folio Sequence</span>
          </div>
          <h1 className="font-serif text-4xl md:text-5xl font-black text-stone-900 tracking-tight leading-none">
            Shakespeare LSTM Text Generator
          </h1>
          <p className="text-stone-500 font-serif italic text-sm mt-1">
            An End-to-End Generative AI sequence model trained on Shakespeare's Complete Works.
          </p>
        </div>

        {/* Global Action Packager */}
        <button
          onClick={handleDownloadCodebase}
          disabled={isZipping}
          id="global-zip-download-btn"
          className="bg-[#78350f] hover:bg-stone-900 disabled:bg-stone-300 text-amber-50 font-serif font-medium py-3 px-6 rounded-xl shadow-md hover:shadow-lg transition-all flex items-center gap-2 text-sm cursor-pointer border border-[#78350f]/20"
        >
          <Download className="w-4 h-4" />
          <span>{isZipping ? "Packaging Codebase..." : "Download Full Python Project"}</span>
        </button>
      </header>

      {/* Quick Specs Dashboard banner */}
      <section className="max-w-7xl mx-auto px-4 py-6" id="global-specs-banner">
        <div className="bg-white border border-stone-200 rounded-2xl p-4 grid grid-cols-2 md:grid-cols-5 gap-4 shadow-sm text-xs font-mono">
          <div className="flex flex-col gap-0.5 pl-2 border-l border-stone-200">
            <span className="text-stone-400 uppercase tracking-widest text-[9px]">Vocab Size</span>
            <span className="font-bold text-stone-800">8,421 Words</span>
          </div>
          <div className="flex flex-col gap-0.5 pl-2 border-l border-stone-200">
            <span className="text-stone-400 uppercase tracking-widest text-[9px]">Layers</span>
            <span className="font-bold text-stone-800">Stacked LSTM</span>
          </div>
          <div className="flex flex-col gap-0.5 pl-2 border-l border-stone-200">
            <span className="text-stone-400 uppercase tracking-widest text-[9px]">Embeddings</span>
            <span className="font-bold text-stone-800">128 Dimensions</span>
          </div>
          <div className="flex flex-col gap-0.5 pl-2 border-l border-stone-200">
            <span className="text-stone-400 uppercase tracking-widest text-[9px]">Loss Function</span>
            <span className="font-bold text-stone-800">Sparse Categorical CE</span>
          </div>
          <div className="flex flex-col gap-0.5 pl-2 border-l border-stone-200 col-span-2 md:col-span-1">
            <span className="text-stone-400 uppercase tracking-widest text-[9px]">Hardware Target</span>
            <span className="font-bold text-emerald-700 flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse"></span>
              GPU Accelerator
            </span>
          </div>
        </div>
      </section>

      {/* Main Navigation tabs */}
      <nav className="max-w-7xl mx-auto px-4 mb-8" id="primary-app-nav">
        <div className="border-b border-stone-200 flex overflow-x-auto gap-1 pb-1 scrollbar-none">
          <button
            onClick={() => setActiveTab("playground")}
            id="tab-playground-btn"
            className={`py-3.5 px-5 text-sm font-serif transition-all flex items-center gap-1.5 border-b-2 cursor-pointer ${
              activeTab === "playground"
                ? "border-[#78350f] text-[#78350f] font-bold"
                : "border-transparent text-stone-500 hover:text-stone-800"
            }`}
          >
            <PlayCircle className="w-4 h-4 flex-shrink-0" />
            <span>🎭 Text Generation</span>
          </button>
          <button
            onClick={() => setActiveTab("sandbox")}
            id="tab-sandbox-btn"
            className={`py-3.5 px-5 text-sm font-serif transition-all flex items-center gap-1.5 border-b-2 cursor-pointer ${
              activeTab === "sandbox"
                ? "border-[#78350f] text-[#78350f] font-bold"
                : "border-transparent text-stone-500 hover:text-stone-800"
            }`}
          >
            <Layers className="w-4 h-4 flex-shrink-0" />
            <span>🔬 Cell Gate Sandbox</span>
          </button>
          <button
            onClick={() => setActiveTab("explorer")}
            id="tab-explorer-btn"
            className={`py-3.5 px-5 text-sm font-serif transition-all flex items-center gap-1.5 border-b-2 cursor-pointer ${
              activeTab === "explorer"
                ? "border-[#78350f] text-[#78350f] font-bold"
                : "border-transparent text-stone-500 hover:text-stone-800"
            }`}
          >
            <Code2 className="w-4 h-4 flex-shrink-0" />
            <span>📂 Code Explorer</span>
          </button>
          <button
            onClick={() => setActiveTab("training")}
            id="tab-training-btn"
            className={`py-3.5 px-5 text-sm font-serif transition-all flex items-center gap-1.5 border-b-2 cursor-pointer ${
              activeTab === "training"
                ? "border-[#78350f] text-[#78350f] font-bold"
                : "border-transparent text-stone-500 hover:text-stone-800"
            }`}
          >
            <ShieldCheck className="w-4 h-4 flex-shrink-0" />
            <span>📈 Training Arena</span>
          </button>
          <button
            onClick={() => setActiveTab("interview")}
            id="tab-interview-btn"
            className={`py-3.5 px-5 text-sm font-serif transition-all flex items-center gap-1.5 border-b-2 cursor-pointer ${
              activeTab === "interview"
                ? "border-[#78350f] text-[#78350f] font-bold"
                : "border-transparent text-stone-500 hover:text-stone-800"
            }`}
          >
            <GraduationCap className="w-4 h-4 flex-shrink-0" />
            <span>🎓 Prep & Quiz</span>
          </button>
        </div>
      </nav>

      {/* Main View Display with transitions */}
      <main className="max-w-7xl mx-auto px-4" id="primary-app-view">
        <AnimatePresence mode="wait">
          <motion.div
            key={activeTab}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.25 }}
          >
            {activeTab === "playground" && <Playground />}
            {activeTab === "sandbox" && <Sandbox />}
            {activeTab === "explorer" && <Explorer />}
            {activeTab === "training" && <Training />}
            {activeTab === "interview" && <Interview />}
          </motion.div>
        </AnimatePresence>
      </main>
    </div>
  );
}
