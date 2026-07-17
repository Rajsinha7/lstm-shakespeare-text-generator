export interface ProjectFile {
  name: string;
  type: "file" | "directory";
  path: string;
  children?: ProjectFile[];
}

export interface AlternativeToken {
  token: string;
  probability: number;
}

export interface TokenDistribution {
  word: string;
  alternatives: AlternativeToken[];
}

export interface BeamSearchCandidate {
  sequence: string;
  score: number;
}

export interface BeamSearchStep {
  step: number;
  candidates: BeamSearchCandidate[];
}

export interface GenerationResponse {
  generatedText: string;
  tokenDistributions: TokenDistribution[];
  beamSearchSteps?: BeamSearchStep[];
}

export interface QuizQuestion {
  id: number;
  question: string;
  options: string[];
  correctAnswer: number;
  explanation: string;
}

export interface TrainingHistoryStep {
  epoch: number;
  loss: number;
  valLoss: number;
  accuracy: number;
  valAccuracy: number;
  perplexity: number;
}
