import express from "express";
import path from "path";
import { exec } from "child_process";
import fs from "fs";
import { createServer as createViteServer } from "vite";
import dotenv from "dotenv";

dotenv.config();

const app = express();
const PORT = 3000;

app.use(express.json());

// API Route: Forward Shakespeare LSTM Text Generation to Python FastAPI Backend
app.post("/api/generate", async (req, res) => {
  const {
    seedText = "Shall I compare thee to a summer's day",
    length = 20,
    temperature = 0.5,
    method = "Temperature Sampling", // "Temperature Sampling" or "Beam Search"
    beamWidth = 3,
  } = req.body;

  // Map React UI parameters to FastAPI snake_case parameters
  const fastapiPayload = {
    seed_text: seedText,
    temperature: temperature,
    next_words: length,
    strategy: method === "Beam Search" ? "beam" : "temperature",
    beam_width: beamWidth
  };

  try {
    const response = await fetch("http://localhost:8000/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(fastapiPayload),
    });

    if (!response.ok) {
      const errText = await response.text();
      let errJson;
      try {
        errJson = JSON.parse(errText);
      } catch {
        errJson = { detail: errText };
      }
      return res.status(response.status).json({
        error: errJson.detail || errJson.error || "Inference backend failed."
      });
    }

    const data: any = await response.json();

    // Map FastAPI snake_case response back to React UI camelCase response
    res.json({
      generatedText: data.generated_text,
      tokenDistributions: data.probabilities,
      beamSearchSteps: data.beam_steps || []
    });
  } catch (err: any) {
    console.error("FastAPI Backend Connection Error:", err);
    res.status(503).json({
      error: `Python backend connection error: ${err.message}. Ensure uvicorn is running.`
    });
  }
});

// API Route: Package and Download python codebase as a zip
app.get("/api/download-zip", (req, res) => {
  const zipPath = path.join(process.cwd(), "shakespeare_lstm_project.zip");
  
  // Use the Python utils script to package the project, which is cross-platform!
  const zipCommand = `python project/src/utils.py`;

  exec(zipCommand, (error) => {
    if (error) {
      console.error("Zipping Command Error:", error);
      return res.status(500).json({ error: "Failed to compress the Python project workspace." });
    }

    if (!fs.existsSync(zipPath)) {
      return res.status(404).json({ error: "Compressed zip was not generated." });
    }

    res.download(zipPath, "shakespeare_lstm_project.zip", (err) => {
      if (err) {
        console.error("Download delivery error:", err);
      }
      // Clean up the temp zip file after transfer completes
      try {
        fs.unlinkSync(zipPath);
      } catch (cleanupErr) {
        console.error("Cleanup error:", cleanupErr);
      }
    });
  });
});

// Serve project files for Code Explorer
app.get("/api/project-files", (req, res) => {
  const getFilesRecursively = (dir: string): any[] => {
    const results: any[] = [];
    if (!fs.existsSync(dir)) return [];
    
    const list = fs.readdirSync(dir);
    list.forEach((file) => {
      const filePath = path.join(dir, file);
      const stat = fs.statSync(filePath);
      if (stat && stat.isDirectory()) {
        if (file !== "venv" && file !== "__pycache__" && file !== "node_modules") {
          results.push({
            name: file,
            type: "directory",
            children: getFilesRecursively(filePath),
            path: path.relative(process.cwd(), filePath),
          });
        }
      } else {
        const ext = path.extname(file);
        if (ext !== ".zip" && ext !== ".pyc" && ext !== ".h5" && ext !== ".pkl") {
          results.push({
            name: file,
            type: "file",
            path: path.relative(process.cwd(), filePath),
          });
        }
      }
    });
    return results;
  };

  try {
    const fileTree = getFilesRecursively(path.join(process.cwd(), "project"));
    res.json(fileTree);
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// Serve single file content
app.get("/api/project-file-content", (req, res) => {
  const filePathQuery = req.query.path as string;
  if (!filePathQuery) return res.status(400).json({ error: "Path query required" });

  const absolutePath = path.resolve(process.cwd(), filePathQuery);
  if (!absolutePath.startsWith(path.resolve(process.cwd(), "project"))) {
    return res.status(403).json({ error: "Access Denied: Path resides outside workspace." });
  }

  try {
    const content = fs.readFileSync(absolutePath, "utf-8");
    res.json({ content });
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// Setup Vite Dev Server / Static Hosting Middleware
async function startServer() {
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on http://localhost:${PORT}`);
  });
}

startServer();
