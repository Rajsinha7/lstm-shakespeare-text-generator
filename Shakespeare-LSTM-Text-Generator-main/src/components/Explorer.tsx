import { useState, useEffect } from "react";
import { Folder, File, Code, Copy, Check, Download, Info, Terminal, FileCode } from "lucide-react";
import { ProjectFile } from "../types";

export default function Explorer() {
  const [files, setFiles] = useState<ProjectFile[]>([]);
  const [selectedFilePath, setSelectedFilePath] = useState<string | null>(null);
  const [fileContent, setFileContent] = useState<string | null>(null);
  const [loadingContent, setLoadingContent] = useState(false);
  const [copied, setCopied] = useState(false);
  const [downloading, setDownloading] = useState(false);

  useEffect(() => {
    fetchFiles();
  }, []);

  const fetchFiles = async () => {
    try {
      const response = await fetch("/api/project-files");
      if (response.ok) {
        const data = await response.json();
        setFiles(data);
        // Automatically open preprocess.py by default if available
        const defaultFile = findFileInTree(data, "preprocess.py");
        if (defaultFile) {
          handleFileSelect(defaultFile.path);
        } else if (data.length > 0) {
          // Fallback to first file found
          const first = findFirstFile(data);
          if (first) handleFileSelect(first.path);
        }
      }
    } catch (err) {
      console.error("Failed to load file index:", err);
    }
  };

  const findFileInTree = (tree: ProjectFile[], name: string): ProjectFile | null => {
    for (const node of tree) {
      if (node.type === "file" && node.name === name) return node;
      if (node.type === "directory" && node.children) {
        const found = findFileInTree(node.children, name);
        if (found) return found;
      }
    }
    return null;
  };

  const findFirstFile = (tree: ProjectFile[]): ProjectFile | null => {
    for (const node of tree) {
      if (node.type === "file") return node;
      if (node.type === "directory" && node.children) {
        const found = findFirstFile(node.children);
        if (found) return found;
      }
    }
    return null;
  };

  const handleFileSelect = async (path: string) => {
    setSelectedFilePath(path);
    setLoadingContent(true);
    setCopied(false);
    try {
      const response = await fetch(`/api/project-file-content?path=${encodeURIComponent(path)}`);
      if (response.ok) {
        const data = await response.json();
        setFileContent(data.content);
      } else {
        setFileContent("Error: Unable to retrieve file contents.");
      }
    } catch (err) {
      setFileContent("Error: Network failure loading content.");
    } finally {
      setLoadingContent(false);
    }
  };

  const handleCopy = () => {
    if (fileContent) {
      navigator.clipboard.writeText(fileContent);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const handleDownloadZip = async () => {
    setDownloading(true);
    try {
      window.location.href = "/api/download-zip";
    } catch (err) {
      console.error(err);
    } finally {
      setTimeout(() => setDownloading(false), 3000);
    }
  };

  const renderFileNode = (node: ProjectFile, depth = 0) => {
    const isDir = node.type === "directory";
    const isSelected = selectedFilePath === node.path;

    return (
      <div key={node.path} style={{ paddingLeft: `${depth * 14}px` }} className="flex flex-col select-none">
        <button
          onClick={() => !isDir && handleFileSelect(node.path)}
          className={`flex items-center gap-2 py-1.5 px-2.5 rounded-lg text-left text-xs transition-colors cursor-pointer w-full ${
            isDir
              ? "text-stone-700 hover:bg-stone-100 font-semibold"
              : isSelected
              ? "bg-amber-100 text-amber-950 font-medium"
              : "text-stone-600 hover:bg-stone-50"
          }`}
        >
          {isDir ? (
            <Folder className="w-4 h-4 text-amber-700 fill-amber-700/20 flex-shrink-0" />
          ) : (
            <FileCode className="w-4 h-4 text-stone-500 flex-shrink-0" />
          )}
          <span className="truncate">{node.name}</span>
        </button>
        {isDir && node.children && node.children.map((child) => renderFileNode(child, depth + 1))}
      </div>
    );
  };

  // Human-readable file explanations
  const getFileExplanation = (pathStr: string | null) => {
    if (!pathStr) return "";
    const filename = pathStr.split("/").pop();
    switch (filename) {
      case "preprocess.py":
        return "Preprocesses character inputs to lowercase, cleans punctuations, tokenizes into space-separated word-level tokens, sliding-windows sequences, and structures training arrays.";
      case "train.py":
        return "Compiles the deep sequence model with custom Embeddings, stacked LSTM blocks, Batch Normalization, and Dropout. Sets up checkpoints, TensorBoard, and outputs training loss logs.";
      case "generate.py":
        return "Core generation inference loops. Implements both Temperature-based multinomial distribution scaling and global joint-probability Beam Search paths.";
      case "utils.py":
        return "Hosts helper metrics like Exponential Perplexity computations, sliding sequence debuggers, dataset coverage calculators, and codebase packagers.";
      case "app.py":
        return "Launches a bonus local Streamlit application presenting simple widgets for in-browser interactive Shakespeare generation.";
      case "requirements.txt":
        return "Declares the exact python dependencies for running the model locally, such as TensorFlow, NumPy, Matplotlib, and Streamlit.";
      case "README.md":
        return "Complete technical guide presenting architectural specifications, mathematical definitions, deployment configurations, and interview preparation QA sheets.";
      case "shakespeare_lstm.ipynb":
        return "A complete, modular Jupyter Notebook formatted for seamless loading and execution in Google Colab using high-speed cloud GPUs.";
      default:
        return "Source code file for the LSTM sequence modeling project.";
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-8" id="code-explorer-container">
      {/* File Tree Sidebar: 4 cols */}
      <div className="lg:col-span-4 bg-white border border-stone-200 rounded-2xl p-5 shadow-sm flex flex-col gap-5 h-fit lg:sticky lg:top-4">
        <div className="flex justify-between items-center border-b border-stone-100 pb-3">
          <div>
            <h4 className="font-serif text-sm font-semibold text-stone-900 flex items-center gap-1.5">
              <Terminal className="w-4 h-4 text-amber-700" />
              Source Explorer
            </h4>
            <p className="text-stone-400 text-[10px]">Modulated workspace directory structure.</p>
          </div>
          <button
            onClick={handleDownloadZip}
            disabled={downloading}
            id="download-project-zip-btn"
            className="p-2 bg-amber-50 hover:bg-amber-100 border border-amber-200 text-amber-900 rounded-lg text-[10px] font-semibold flex items-center gap-1 transition-colors cursor-pointer"
            title="Download full codebase as a ZIP file"
          >
            <Download className="w-3.5 h-3.5" />
            <span>{downloading ? "Archiving..." : "Download ZIP"}</span>
          </button>
        </div>

        <div className="flex flex-col gap-1.5 max-h-[400px] overflow-y-auto pr-1">
          {files.map((file) => renderFileNode(file))}
        </div>

        {/* Selected file info card */}
        {selectedFilePath && (
          <div className="bg-[#faf9f6] border border-[#e7e5e4] p-3.5 rounded-xl flex items-start gap-2.5 text-xs">
            <Info className="w-4 h-4 text-amber-700 mt-0.5 flex-shrink-0" />
            <div>
              <span className="font-semibold text-stone-800 block mb-0.5">{selectedFilePath.split("/").pop()}</span>
              <p className="text-stone-500 leading-relaxed text-[11px]">{getFileExplanation(selectedFilePath)}</p>
            </div>
          </div>
        )}
      </div>

      {/* Code Viewer Panel: 8 cols */}
      <div className="lg:col-span-8 flex flex-col gap-4">
        <div className="bg-white border border-stone-200 rounded-2xl shadow-sm overflow-hidden flex flex-col">
          <div className="bg-stone-900 px-5 py-3.5 flex justify-between items-center border-b border-stone-800 flex-wrap gap-2">
            <span className="text-stone-300 font-mono text-xs flex items-center gap-2">
              <Code className="w-4 h-4 text-amber-500" />
              {selectedFilePath || "No file selected"}
            </span>
            {fileContent && (
              <button
                onClick={handleCopy}
                id="copy-code-to-clipboard-btn"
                className="text-stone-400 hover:text-white transition-colors text-xs flex items-center gap-1.5 font-mono px-2.5 py-1 rounded bg-stone-800 border border-stone-700 cursor-pointer"
              >
                {copied ? (
                  <>
                    <Check className="w-3.5 h-3.5 text-emerald-500" />
                    <span>Copied!</span>
                  </>
                ) : (
                  <>
                    <Copy className="w-3.5 h-3.5" />
                    <span>Copy Code</span>
                  </>
                )}
              </button>
            )}
          </div>

          <div className="p-4 bg-stone-950 max-h-[500px] overflow-y-auto custom-scrollbar font-mono text-xs text-stone-300 leading-relaxed border-t border-stone-900">
            {loadingContent ? (
              <div id="code-loading-state" className="py-24 text-center text-stone-500 flex flex-col items-center justify-center gap-2">
                <div className="w-6 h-6 border-2 border-amber-600 border-t-transparent rounded-full animate-spin" />
                <span>Streaming lines...</span>
              </div>
            ) : fileContent ? (
              <pre id="code-content-display" className="whitespace-pre overflow-x-auto pr-2">
                {fileContent}
              </pre>
            ) : (
              <div className="py-24 text-center text-stone-500 font-serif italic">
                Select a Python script from the browser sidebar tree to inspect details.
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
