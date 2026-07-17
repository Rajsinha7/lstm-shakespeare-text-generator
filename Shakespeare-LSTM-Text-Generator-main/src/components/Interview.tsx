import { useState } from "react";
import { GraduationCap, BookOpen, CheckCircle, XCircle, ChevronRight, HelpCircle, AlertCircle, Award } from "lucide-react";
import { QuizQuestion } from "../types";

const STUDY_GUIDES = [
  {
    topic: "LSTMs vs. Transformers",
    summary: "Transformers process entire sequences simultaneously using Multi-Head Self-Attention, yielding O(N^2) complexity but allowing infinite parallelization on GPUs. LSTMs process text sequentially (O(N) complexity) passing hidden states step-by-step. While Transformers dominate massive language models, LSTMs remain highly efficient for moderate sequence scales, embedded systems, and resource-constrained environments.",
    point1: "Parallelization: Transformers are parallel across tokens; LSTMs are strictly sequential.",
    point2: "Path Length: Transformers connect any two tokens in O(1) distance; LSTMs suffer from sequential information decay."
  },
  {
    topic: "The Vanishing Gradient Remedy",
    summary: "Vanilla RNNs multiply gradients back through the sequence via recurrent weight matrices, causing gradients to vanish or explode exponentially. LSTMs bypass this recursive multiplication by introducing the cell state (C_t) conveyor belt. Updates to C_t are additive, controlled by gating coefficients. This linear pathway allows error signals to flow back hundreds of steps without decaying.",
    point1: "Additive Updates: C_t updates linearly, bypassing recursive matrix multiplications.",
    point2: "Gating Networks: Sigmoid gates regulate precisely how much error signal flows backward."
  },
  {
    topic: "Word Embeddings & Semantic Coordinate Spaces",
    summary: "One-hot encodings treat words as orthogonal unit vectors, completely ignoring semantic relationships. An Embedding layer maps each vocabulary index to a dense, continuous, low-dimensional coordinate vector (e.g., 128 dimensions). During backpropagation, these coordinates adjust so that words appearing in similar contexts end up clustered together, letting the model learn abstract semantic similarities.",
    point1: "Dimensionality: Condenses sparse vectors (e.g., 10,000d) into dense, low-dimensional spaces (e.g., 128d).",
    point2: "Context Learning: Words with similar semantic roles align dynamically in coordinate space."
  }
];

const QUIZ_QUESTIONS: QuizQuestion[] = [
  {
    id: 1,
    question: "Why does the LSTM architecture successfully prevent the vanishing gradient problem compared to a vanilla Recurrent Neural Network (RNN)?",
    options: [
      "Because it replaces backpropagation with feed-forward reinforcement loops.",
      "Because it updates its Cell State through linear addition, allowing gradients to propagate backward with minimal decay.",
      "Because it utilizes a Softmax activation layer that bounds weight values.",
      "Because its embedding weights are static and do not require gradient updates."
    ],
    correctAnswer: 1,
    explanation: "Correct! Vanilla RNNs multiply gradients recursively through time-steps, leading to exponential decay. LSTMs maintain an internal Cell State updated through linear addition (scaled by sigmoid gates), which allows the gradient error to propagate backward across long steps with minimal decay."
  },
  {
    id: 2,
    question: "When predicting the next word, what is the mathematical effect of setting a very low temperature (e.g. T = 0.2) during Softmax sampling?",
    options: [
      "It makes the probability distribution uniform, resulting in completely random predictions.",
      "It increases the cumulative joint log-likelihood of all candidates equally.",
      "It sharpens the probability distribution, making the most likely token extremely dominant and predictions highly deterministic.",
      "It disables the forget gate, forcing the model to repeat the initial seed text indefinitely."
    ],
    correctAnswer: 2,
    explanation: "Correct! Dividing logits by a low temperature (T < 1) scales the differences between values. When softmax is applied, the largest logit becomes highly dominant (approaching 100%), making predictions safe, confident, and highly repetitive."
  },
  {
    id: 3,
    question: "In word-level sequence modeling, how does Beam Search differ from standard Greedy Search (Argmax)?",
    options: [
      "Greedy Search evaluates all vocabulary permutations simultaneously, while Beam Search is restricted to nouns.",
      "Beam Search maintains the top K high-probability sequence paths at each step, preventing short-sighted local decisions.",
      "Greedy Search uses deep neural layers, whereas Beam Search runs purely on statistical counts.",
      "Beam Search requires character-level tokenization to function correctly."
    ],
    correctAnswer: 1,
    explanation: "Correct! Greedy search selects only the single highest-probability token at each step (width=1), which can lead to poor choices if a high-probability word leads to nonsense. Beam Search tracks the top K paths, evaluating cumulative joint probabilities across the entire generated block."
  },
  {
    id: 4,
    question: "What is the computational complexity of the Self-Attention mechanism in a Transformer with respect to sequence length N, and how does it compare to an LSTM?",
    options: [
      "Transformer attention is O(N) while LSTM is O(N^2).",
      "Both Transformer attention and LSTMs operate at O(1) constant complexity.",
      "Transformer attention is O(N^2) due to pairwise token comparisons, while LSTM recurrent operations are O(N).",
      "Transformer attention is O(log N) while LSTM is O(N log N)."
    ],
    correctAnswer: 2,
    explanation: "Correct! Self-attention computes pairwise compatibility scores between every token in the sequence, resulting in an O(N^2) computational complexity. In contrast, an LSTM processes tokens sequentially, requiring O(N) operations."
  }
];

export default function Interview() {
  const [quizStarted, setQuizStarted] = useState(false);
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [selectedAnswer, setSelectedAnswer] = useState<number | null>(null);
  const [quizScore, setQuizScore] = useState(0);
  const [quizFinished, setQuizFinished] = useState(false);

  const handleAnswerSelect = (optionIndex: number) => {
    if (selectedAnswer !== null) return;
    setSelectedAnswer(optionIndex);
    if (optionIndex === QUIZ_QUESTIONS[currentQuestionIndex].correctAnswer) {
      setQuizScore(quizScore + 1);
    }
  };

  const handleNext = () => {
    setSelectedAnswer(null);
    if (currentQuestionIndex < QUIZ_QUESTIONS.length - 1) {
      setCurrentQuestionIndex(currentQuestionIndex + 1);
    } else {
      setQuizFinished(true);
    }
  };

  const restartQuiz = () => {
    setQuizStarted(true);
    setCurrentQuestionIndex(0);
    setSelectedAnswer(null);
    setQuizScore(0);
    setQuizFinished(false);
  };

  const activeQuestion = QUIZ_QUESTIONS[currentQuestionIndex];

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-8" id="interview-hub-container">
      {/* Study Guides: 6 cols */}
      <div className="lg:col-span-6 flex flex-col gap-6">
        <div>
          <h3 className="font-serif text-2xl font-bold text-stone-900 flex items-center gap-2">
            <GraduationCap className="w-6 h-6 text-amber-700" />
            Interview Preparation Guide
          </h3>
          <p className="text-stone-500 text-xs mt-1">
            Master core Recurrent Neural Network theory and sequence modeling concepts.
          </p>
        </div>

        <div className="flex flex-col gap-6 max-h-[550px] overflow-y-auto pr-1">
          {STUDY_GUIDES.map((guide, idx) => (
            <div key={guide.topic} className="bg-white border border-stone-200 rounded-2xl p-5 shadow-sm flex flex-col gap-3" id={`study-guide-${idx}`}>
              <h4 className="font-serif font-bold text-stone-900 text-base flex items-center gap-2">
                <span className="w-5 h-5 bg-amber-50 border border-amber-200 text-amber-900 font-mono text-xs rounded-full flex items-center justify-center font-normal">
                  {idx + 1}
                </span>
                {guide.topic}
              </h4>
              <p className="text-stone-600 text-xs leading-relaxed">{guide.summary}</p>
              
              <div className="border-t border-stone-100 pt-3 flex flex-col gap-1.5 text-xs text-stone-500">
                <span className="font-semibold text-stone-700 block uppercase tracking-wider text-[9px]">Core Cheat Sheet Details:</span>
                <span className="flex items-start gap-1.5">
                  <span className="text-amber-600 mt-0.5">•</span>
                  <span>{guide.point1}</span>
                </span>
                <span className="flex items-start gap-1.5">
                  <span className="text-amber-600 mt-0.5">•</span>
                  <span>{guide.point2}</span>
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Interactive Quiz Module: 6 cols */}
      <div className="lg:col-span-6 flex flex-col gap-6">
        <div>
          <h3 className="font-serif text-2xl font-bold text-stone-900 flex items-center gap-2">
            <HelpCircle className="w-6 h-6 text-amber-700" />
            Self-Assessment Quiz
          </h3>
          <p className="text-stone-500 text-xs mt-1">
            Test your sequence modeling knowledge with detailed feedback.
          </p>
        </div>

        <div className="bg-white border border-stone-200 rounded-2xl p-6 shadow-sm min-h-[420px] flex flex-col justify-between" id="interactive-quiz-box">
          {!quizStarted ? (
            <div className="text-center py-12 flex flex-col items-center justify-center gap-4 flex-grow">
              <Award className="w-16 h-16 text-amber-100 stroke-[1.5]" />
              <h4 className="font-serif text-lg font-bold text-stone-900">LSTM & NLP Readiness Check</h4>
              <p className="text-stone-500 text-xs max-w-sm leading-relaxed">
                A 4-question technical check assessing your readiness for advanced deep learning sequence-modeling internships and coding reviews.
              </p>
              <button
                onClick={restartQuiz}
                id="start-quiz-btn"
                className="bg-[#78350f] hover:bg-stone-900 text-amber-50 font-serif font-medium py-2.5 px-6 rounded-xl transition-all shadow cursor-pointer mt-2"
              >
                Launch Technical Quiz
              </button>
            </div>
          ) : quizFinished ? (
            <div className="text-center py-12 flex flex-col items-center justify-center gap-4 flex-grow">
              <CheckCircle className="w-16 h-16 text-emerald-100 stroke-[1.5]" />
              <h4 className="font-serif text-xl font-bold text-stone-900">Evaluation Finished!</h4>
              <p className="text-stone-500 text-xs max-w-sm">
                You successfully completed the NLP Technical Quiz with a score of:
              </p>
              <span className="text-4xl font-mono text-stone-800 font-bold bg-stone-100 px-6 py-2 rounded-xl border border-stone-200" id="quiz-final-score">
                {quizScore} / {QUIZ_QUESTIONS.length}
              </span>
              <button
                onClick={restartQuiz}
                id="restart-quiz-btn"
                className="bg-stone-900 hover:bg-stone-800 text-white font-serif font-medium py-2.5 px-6 rounded-xl transition-all shadow cursor-pointer mt-2"
              >
                Restart Quiz
              </button>
            </div>
          ) : (
            <div className="flex flex-col justify-between flex-grow" id="active-quiz-question-container">
              <div>
                {/* Header status */}
                <div className="flex justify-between items-center text-xs text-stone-400 font-mono mb-4 border-b border-stone-100 pb-2">
                  <span>QUESTION {currentQuestionIndex + 1} OF {QUIZ_QUESTIONS.length}</span>
                  <span>SCORE: {quizScore}</span>
                </div>

                {/* Question */}
                <h4 className="font-serif text-sm font-semibold text-stone-900 leading-relaxed mb-5" id="quiz-question-text">
                  {activeQuestion.question}
                </h4>

                {/* Options list */}
                <div className="flex flex-col gap-2.5">
                  {activeQuestion.options.map((opt, oIdx) => {
                    const isSelected = selectedAnswer === oIdx;
                    const isCorrect = oIdx === activeQuestion.correctAnswer;
                    const isWrongSelected = isSelected && !isCorrect;

                    let btnClass = "border-stone-200 hover:bg-stone-50 text-stone-700 bg-white";
                    if (selectedAnswer !== null) {
                      if (isCorrect) btnClass = "bg-emerald-50 border-emerald-500 text-emerald-950 font-medium";
                      else if (isWrongSelected) btnClass = "bg-red-50 border-red-500 text-red-950";
                      else btnClass = "opacity-50 border-stone-100 text-stone-400";
                    }

                    return (
                      <button
                        key={oIdx}
                        id={`quiz-option-btn-${oIdx}`}
                        onClick={() => handleAnswerSelect(oIdx)}
                        disabled={selectedAnswer !== null}
                        className={`text-left p-3.5 text-xs rounded-xl border transition-all flex items-start gap-2.5 ${btnClass} ${selectedAnswer === null ? "cursor-pointer" : "cursor-default"}`}
                      >
                        <span className="font-mono text-stone-400 mt-0.5">{String.fromCharCode(65 + oIdx)}.</span>
                        <span className="leading-relaxed">{opt}</span>
                      </button>
                    );
                  })}
                </div>
              </div>

              {/* Explanation section if answered */}
              {selectedAnswer !== null && (
                <div className="mt-6 flex flex-col gap-3" id="quiz-explanation-box">
                  <div className={`p-4 rounded-xl text-xs leading-relaxed ${
                    selectedAnswer === activeQuestion.correctAnswer 
                      ? "bg-emerald-50 text-emerald-800 border border-emerald-200" 
                      : "bg-red-50 text-red-800 border border-red-200"
                  }`}>
                    <span className="font-bold flex items-center gap-1 mb-1">
                      {selectedAnswer === activeQuestion.correctAnswer ? (
                        <>
                          <CheckCircle className="w-4 h-4 text-emerald-700" />
                          Excellent!
                        </>
                      ) : (
                        <>
                          <XCircle className="w-4 h-4 text-red-700" />
                          Review Concept
                        </>
                      )}
                    </span>
                    <p>{activeQuestion.explanation}</p>
                  </div>

                  <button
                    onClick={handleNext}
                    id="next-quiz-question-btn"
                    className="self-end bg-stone-900 hover:bg-stone-800 text-white font-mono text-xs font-semibold py-2 px-4 rounded-lg flex items-center gap-1 cursor-pointer transition-colors"
                  >
                    <span>{currentQuestionIndex === QUIZ_QUESTIONS.length - 1 ? "Finish Quiz" : "Next Question"}</span>
                    <ChevronRight className="w-4 h-4" />
                  </button>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
