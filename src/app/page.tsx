"use client"

import { useEffect, useMemo, useState } from "react"
import {
  AlertOctagon,
  BarChart3,
  Brain,
  CheckCircle2,
  Database,
  FileWarning,
  Loader2,
  MessageSquareText,
  Play,
  Shield,
  Sliders,
  Sparkles,
  Terminal,
  Trash2,
} from "lucide-react"

type ModelExport = {
  model_type: string
  positive_label: string
  negative_label: string
  thresholds: { allow: number; review: number }
  vocabulary: Record<string, number>
  weights: {
    dense1_kernel: number[][]
    dense1_bias: number[]
    dense2_kernel: number[][]
    dense2_bias: number[]
    out_kernel: number[][]
    out_bias: number[]
  }
  metrics: {
    dataset: string
    train_size: number
    validation_size: number
    test_size: number
    vocabulary_size: number
    accuracy: number
    precision: number
    recall: number
    f1: number
    epochs_ran: number
  }
}

type ModerationAction = "allow" | "review" | "block"

const presets = [
  {
    title: "Phishing / Prize Scam",
    category: "High Risk",
    text: "URGENT! Your account won a $1,000 cash prize. Call 09061701461 now or click the verification link to claim your free reward today.",
  },
  {
    title: "Grey-Zone Promo",
    category: "Borderline",
    text: "Special limited offer for existing subscribers: upgrade today for a free bonus month and priority support access.",
  },
  {
    title: "Engineering Standup Note",
    category: "Clean",
    text: "Hey team, I pushed the PyTorch ECG evaluation metrics and updated the FastAPI Docker container. Let's review the PR at 14:00.",
  },
  {
    title: "Subscription Support",
    category: "Clean",
    text: "Hello, could you please send me the PDF invoice for our team seats from last month? Thanks!",
  },
]

function tokenize(text: string): string[] {
  return text.toLowerCase().match(/[a-z0-9']+/g) || []
}

function vectorize(text: string, vocabulary: Record<string, number>): { vector: number[]; matched: string[] } {
  const size = Object.keys(vocabulary).length
  const vector = new Array<number>(size).fill(0)
  const matched = new Set<string>()

  for (const token of tokenize(text)) {
    const index = vocabulary[token]
    if (index !== undefined) {
      vector[index] += 1
      matched.add(token)
    }
  }

  const total = vector.reduce((sum, value) => sum + value, 0)
  const scaled = vector.map((value) => (total > 0 ? Math.log1p(value) / Math.log1p(total) : 0))
  return { vector: scaled, matched: Array.from(matched) }
}

function denseRelu(input: number[], kernel: number[][], bias: number[]): number[] {
  return bias.map((b, outIndex) => {
    let sum = b
    for (let inIndex = 0; inIndex < input.length; inIndex += 1) {
      sum += input[inIndex] * kernel[inIndex][outIndex]
    }
    return Math.max(0, sum)
  })
}

function denseSigmoid(input: number[], kernel: number[][], bias: number[]): number {
  let sum = bias[0] || 0
  for (let inIndex = 0; inIndex < input.length; inIndex += 1) {
    sum += input[inIndex] * kernel[inIndex][0]
  }
  return 1 / (1 + Math.exp(-sum))
}

function scoreTokenImpacts(tokens: string[], model: ModelExport): { token: string; delta: number }[] {
  const unique = Array.from(new Set(tokens)).slice(0, 24)
  return unique
    .map((tok) => {
      const { vector } = vectorize(tok, model.vocabulary)
      const h1 = denseRelu(vector, model.weights.dense1_kernel, model.weights.dense1_bias)
      const h2 = denseRelu(h1, model.weights.dense2_kernel, model.weights.dense2_bias)
      const prob = denseSigmoid(h2, model.weights.out_kernel, model.weights.out_bias)
      return { token: tok, delta: prob }
    })
    .sort((a, b) => b.delta - a.delta)
}

export default function Home() {
  const [model, setModel] = useState<ModelExport | null>(null)
  const [loading, setLoading] = useState(true)
  const [text, setText] = useState(presets[0].text)
  const [allowThreshold, setAllowThreshold] = useState(0.25)
  const [reviewThreshold, setReviewThreshold] = useState(0.65)
  const [history, setHistory] = useState<{ text: string; prob: number; action: ModerationAction }[]>([])

  useEffect(() => {
    fetch("/model/moderation_model.json")
      .then((response) => response.json())
      .then((data: ModelExport) => {
        setModel(data)
        if (data.thresholds) {
          setAllowThreshold(data.thresholds.allow)
          setReviewThreshold(data.thresholds.review)
        }
        setLoading(false)
      })
      .catch(() => setLoading(false))
  }, [])

  const result = useMemo(() => {
    if (!model) return null
    const start = performance.now()
    const { vector, matched } = vectorize(text, model.vocabulary)
    const hidden1 = denseRelu(vector, model.weights.dense1_kernel, model.weights.dense1_bias)
    const hidden2 = denseRelu(hidden1, model.weights.dense2_kernel, model.weights.dense2_bias)
    const probability = denseSigmoid(hidden2, model.weights.out_kernel, model.weights.out_bias)
    const latencyMs = (performance.now() - start).toFixed(2)

    const h1Active = hidden1.filter((v) => v > 0).length
    const h2Active = hidden2.filter((v) => v > 0).length

    const action: ModerationAction =
      probability >= reviewThreshold ? "block" : probability >= allowThreshold ? "review" : "allow"

    const tokenScores = scoreTokenImpacts(matched, model)

    return {
      probability,
      action,
      matchedTokens: matched,
      tokenScores,
      latencyMs,
      h1Active,
      h1Total: hidden1.length,
      h2Active,
      h2Total: hidden2.length,
    }
  }, [model, text, allowThreshold, reviewThreshold])

  const probability = result?.probability ?? 0
  const percent = Math.round(probability * 1000) / 10
  const action = result?.action ?? "allow"

  function logCurrentRun() {
    if (!result) return
    setHistory((prev) => [{ text: text.slice(0, 90), prob: result.probability, action: result.action }, ...prev.slice(0, 5)])
  }

  return (
    <main className="min-h-screen bg-[#07050d] text-slate-100 selection:bg-purple-500/30">
      <div className="pointer-events-none fixed inset-0 bg-[radial-gradient(circle_at_15%_15%,rgba(168,85,247,0.14),transparent_35%),radial-gradient(circle_at_85%_80%,rgba(236,72,153,0.1),transparent_40%)]" />

      <header className="relative border-b border-purple-500/20 bg-[#0b0716]/80 backdrop-blur-md">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-4 px-6 py-4">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl border border-purple-500/40 bg-gradient-to-br from-purple-600/30 to-fuchsia-600/20 text-purple-300 shadow-lg shadow-purple-950/50">
              <Shield className="h-5 w-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="text-lg font-black tracking-tight text-white">GuardRail Neural Inspector</span>
                <span className="rounded-full border border-purple-500/30 bg-purple-950/60 px-2.5 py-0.5 font-mono text-[11px] font-semibold text-purple-300">
                  Keras Tensor Engine
                </span>
              </div>
              <p className="text-xs text-slate-400">
                In-Browser Dense(96) → Dense(32) → Sigmoid Inference · Trained on UCI SMS Spam Corpus
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3 text-xs font-mono">
            <span className="inline-flex items-center gap-1.5 rounded-lg border border-emerald-500/30 bg-emerald-950/40 px-3 py-1.5 text-emerald-300">
              <span className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
              {loading ? "Loading weights..." : `Weights Ready · ${result?.latencyMs || "0.15"} ms/forward`}
            </span>
          </div>
        </div>
      </header>

      <section className="relative mx-auto grid max-w-7xl gap-6 px-6 py-8 lg:grid-cols-[420px_1fr]">
        {/* Left Control Column */}
        <aside className="space-y-5">
          <div className="rounded-2xl border border-purple-500/25 bg-[#0e091d]/90 p-5 shadow-2xl shadow-black/50 backdrop-blur-sm">
            <div className="mb-3 flex items-center justify-between">
              <span className="flex items-center gap-2 text-xs font-mono uppercase tracking-wider text-purple-300">
                <MessageSquareText className="h-4 w-4 text-purple-400" />
                Live Payload Input
              </span>
              <button
                onClick={() => setText("")}
                className="inline-flex items-center gap-1 text-xs text-slate-400 hover:text-white"
              >
                <Trash2 className="h-3.5 w-3.5" /> Clear
              </button>
            </div>

            <textarea
              value={text}
              onChange={(event) => setText(event.target.value)}
              className="h-44 w-full resize-none rounded-xl border border-purple-500/25 bg-[#07050d] p-4 font-mono text-sm leading-relaxed text-slate-100 outline-none transition focus:border-purple-400"
              placeholder="Paste any SMS, support message, or user comment to run real-time neural moderation..."
            />

            <div className="mt-3 flex items-center justify-between text-xs font-mono text-slate-400">
              <span>Tokens: {tokenize(text).length}</span>
              <span>Vocab hits: {result?.matchedTokens.length ?? 0} / {model?.metrics.vocabulary_size ?? 1600}</span>
            </div>

            <button
              onClick={logCurrentRun}
              disabled={!model || !text.trim()}
              className="mt-4 flex w-full items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-purple-500 to-fuchsia-500 px-4 py-3 text-sm font-bold text-white shadow-lg shadow-purple-500/25 transition hover:from-purple-600 hover:to-fuchsia-600 disabled:opacity-50"
            >
              {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
              Snapshot & Log Audit Decision
            </button>
          </div>

          {/* Preset Test Payloads */}
          <div className="rounded-2xl border border-purple-500/20 bg-[#0e091d]/90 p-5">
            <div className="mb-3 text-xs font-mono uppercase tracking-wider text-purple-300">
              Benchmark Test Vectors
            </div>
            <div className="space-y-2">
              {presets.map((item) => (
                <button
                  key={item.title}
                  onClick={() => setText(item.text)}
                  className={`w-full rounded-xl border p-3 text-left transition ${
                    text === item.text
                      ? "border-purple-500/60 bg-purple-950/50"
                      : "border-purple-500/15 bg-[#090612] hover:border-purple-500/35"
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-bold text-white">{item.title}</span>
                    <span className="rounded-md bg-purple-950/80 px-2 py-0.5 font-mono text-[10px] text-purple-300">
                      {item.category}
                    </span>
                  </div>
                  <p className="mt-1 line-clamp-2 text-xs text-slate-400">{item.text}</p>
                </button>
              ))}
            </div>
          </div>

          {/* Policy Threshold Calibration */}
          <div className="rounded-2xl border border-purple-500/20 bg-[#0e091d]/90 p-5">
            <div className="mb-4 flex items-center gap-2 text-xs font-mono uppercase tracking-wider text-purple-300">
              <Sliders className="h-4 w-4 text-purple-400" />
              Policy Decision Thresholds
            </div>

            <div className="space-y-4 text-xs">
              <div>
                <div className="mb-1 flex justify-between">
                  <span className="text-slate-300">Review Gate Threshold</span>
                  <span className="font-mono font-bold text-amber-300">{Math.round(allowThreshold * 100)}%</span>
                </div>
                <input
                  type="range"
                  min={0.05}
                  max={0.5}
                  step={0.01}
                  value={allowThreshold}
                  onChange={(e) => setAllowThreshold(Number(e.target.value))}
                  className="w-full accent-purple-500"
                />
              </div>

              <div>
                <div className="mb-1 flex justify-between">
                  <span className="text-slate-300">Auto-Block Threshold</span>
                  <span className="font-mono font-bold text-rose-300">{Math.round(reviewThreshold * 100)}%</span>
                </div>
                <input
                  type="range"
                  min={0.51}
                  max={0.95}
                  step={0.01}
                  value={reviewThreshold}
                  onChange={(e) => setReviewThreshold(Number(e.target.value))}
                  className="w-full accent-fuchsia-500"
                />
              </div>
            </div>
          </div>
        </aside>

        {/* Right Telemetry & Explanation Column */}
        <section className="space-y-6">
          {/* Top Metrics Row */}
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {[
              ["Corpus Benchmark", model?.metrics.dataset || "UCI SMS Spam"],
              ["Held-Out Accuracy", model ? `${(model.metrics.accuracy * 100).toFixed(1)}%` : "..."],
              ["Macro F1 Score", model ? `${(model.metrics.f1 * 100).toFixed(1)}%` : "..."],
              ["Exported Vocab", model ? `${model.metrics.vocabulary_size.toLocaleString()} tokens` : "..."],
            ].map(([label, value]) => (
              <div
                key={label}
                className="rounded-2xl border border-purple-500/20 bg-[#0e091d]/90 p-4 shadow-lg shadow-purple-950/10"
              >
                <div className="text-xs font-mono uppercase tracking-wider text-slate-400">{label}</div>
                <div className="mt-2 text-2xl font-black text-white">{value}</div>
              </div>
            ))}
          </div>

          {/* Primary Decision Banner */}
          <div className="grid gap-6 lg:grid-cols-[1.2fr_0.8fr]">
            <div className="rounded-2xl border border-purple-500/25 bg-[#0e091d]/95 p-6">
              <div className="mb-4 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Brain className="h-5 w-5 text-purple-400" />
                  <h2 className="text-lg font-bold text-white">Real-Time Sigmoid Risk Telemetry</h2>
                </div>
                <span
                  className={`inline-flex items-center gap-1.5 rounded-full px-3.5 py-1 text-xs font-black uppercase tracking-wider ${
                    action === "block"
                      ? "border border-rose-500/40 bg-rose-500/20 text-rose-200"
                      : action === "review"
                        ? "border border-amber-500/40 bg-amber-500/20 text-amber-200"
                        : "border border-emerald-500/40 bg-emerald-500/20 text-emerald-200"
                  }`}
                >
                  {action === "allow" ? <CheckCircle2 className="h-3.5 w-3.5" /> : <AlertOctagon className="h-3.5 w-3.5" />}
                  Policy Action: {action}
                </span>
              </div>

              <div className="rounded-xl border border-purple-500/15 bg-[#080511] p-5">
                <div className="mb-2 flex items-baseline justify-between">
                  <span className="text-xs font-mono uppercase tracking-wider text-slate-400">
                    Unsafe / Spam Posterior P(y=1|x)
                  </span>
                  <span className="font-mono text-3xl font-black text-purple-300">{percent}%</span>
                </div>

                {/* Progress Bar with Threshold Markers */}
                <div className="relative h-4 w-full overflow-hidden rounded-full bg-white/10">
                  <div
                    className={`h-full transition-all duration-300 ${
                      action === "block"
                        ? "bg-gradient-to-r from-amber-500 to-rose-500"
                        : action === "review"
                          ? "bg-gradient-to-r from-purple-500 to-amber-400"
                          : "bg-gradient-to-r from-emerald-500 to-teal-400"
                    }`}
                    style={{ width: `${Math.max(2, Math.min(100, percent))}%` }}
                  />
                </div>

                <div className="mt-2 flex justify-between font-mono text-[11px] text-slate-400">
                  <span>0% (Clean)</span>
                  <span>Review Gate ({Math.round(allowThreshold * 100)}%)</span>
                  <span>Block Gate ({Math.round(reviewThreshold * 100)}%)</span>
                  <span>100%</span>
                </div>

                {/* Layer Activation Telemetry */}
                <div className="mt-5 grid grid-cols-3 gap-3 border-t border-purple-500/15 pt-4 font-mono text-xs">
                  <div className="rounded-lg bg-[#0e091d] p-2.5">
                    <div className="text-[10px] text-slate-400">Dense Layer 1 (ReLU)</div>
                    <div className="mt-1 font-bold text-purple-200">
                      {result?.h1Active ?? 0} / {result?.h1Total ?? 96} active
                    </div>
                  </div>
                  <div className="rounded-lg bg-[#0e091d] p-2.5">
                    <div className="text-[10px] text-slate-400">Dense Layer 2 (ReLU)</div>
                    <div className="mt-1 font-bold text-purple-200">
                      {result?.h2Active ?? 0} / {result?.h2Total ?? 32} active
                    </div>
                  </div>
                  <div className="rounded-lg bg-[#0e091d] p-2.5">
                    <div className="text-[10px] text-slate-400">Sigmoid Logit Gate</div>
                    <div className="mt-1 font-bold text-fuchsia-300">{probability.toFixed(4)}</div>
                  </div>
                </div>
              </div>
            </div>

            {/* Model Artifact Specs */}
            <div className="rounded-2xl border border-purple-500/20 bg-[#0e091d]/90 p-6">
              <div className="mb-4 flex items-center gap-2">
                <BarChart3 className="h-5 w-5 text-fuchsia-400" />
                <h2 className="text-lg font-bold text-white">Evaluation Split Metrics</h2>
              </div>
              {model ? (
                <div className="space-y-2.5 font-mono text-xs">
                  {[
                    ["Train / Val / Test", `${model.metrics.train_size} / ${model.metrics.validation_size} / ${model.metrics.test_size}`],
                    ["Test Precision", `${(model.metrics.precision * 100).toFixed(2)}%`],
                    ["Test Recall", `${(model.metrics.recall * 100).toFixed(2)}%`],
                    ["Training Epochs", String(model.metrics.epochs_ran)],
                    ["Execution Engine", "Pure TypeScript Matrix Ops"],
                  ].map(([label, value]) => (
                    <div
                      key={label}
                      className="flex items-center justify-between rounded-xl border border-purple-500/15 bg-[#080511] px-3.5 py-2.5"
                    >
                      <span className="text-slate-400">{label}</span>
                      <span className="font-bold text-purple-200">{value}</span>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-sm text-slate-400">Loading Keras model artifacts...</div>
              )}
            </div>
          </div>

          {/* Token-Level Attribution & Audit Log */}
          <div className="grid gap-6 lg:grid-cols-[1.2fr_0.8fr]">
            <div className="rounded-2xl border border-purple-500/20 bg-[#0e091d]/90 p-6">
              <div className="mb-2 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Sparkles className="h-5 w-5 text-purple-400" />
                  <h2 className="text-lg font-bold text-white">Token-Level Risk Attribution</h2>
                </div>
                <span className="font-mono text-xs text-slate-400">Single-Token Marginal Sigmoid Probe</span>
              </div>
              <p className="mb-4 text-xs text-slate-400">
                Each matched vocabulary token is probed individually through the exported Keras weights to rank its contribution toward the spam/unsafe class:
              </p>

              {result && result.tokenScores.length > 0 ? (
                <div className="grid gap-2.5 sm:grid-cols-2">
                  {result.tokenScores.slice(0, 10).map((item) => {
                    const riskPct = Math.round(item.delta * 100)
                    const high = riskPct >= 45
                    return (
                      <div
                        key={item.token}
                        className="flex items-center justify-between rounded-xl border border-purple-500/15 bg-[#080511] px-3.5 py-2"
                      >
                        <span className="font-mono text-xs font-semibold text-white">"{item.token}"</span>
                        <div className="flex items-center gap-2">
                          <div className="h-1.5 w-16 overflow-hidden rounded-full bg-white/10">
                            <div
                              className={`h-full ${high ? "bg-rose-400" : "bg-purple-400"}`}
                              style={{ width: `${Math.max(6, riskPct)}%` }}
                            />
                          </div>
                          <span className={`font-mono text-xs font-bold ${high ? "text-rose-300" : "text-purple-300"}`}>
                            {riskPct}%
                          </span>
                        </div>
                      </div>
                    )
                  })}
                </div>
              ) : (
                <div className="rounded-xl border border-purple-500/15 bg-[#080511] p-4 text-xs text-slate-400">
                  No known vocabulary tokens found in current input.
                </div>
              )}
            </div>

            <div className="rounded-2xl border border-purple-500/20 bg-[#0e091d]/90 p-6">
              <div className="mb-4 flex items-center gap-2">
                <Terminal className="h-5 w-5 text-purple-400" />
                <h2 className="text-lg font-bold text-white">Recent Audit Snapshots</h2>
              </div>

              {history.length > 0 ? (
                <div className="space-y-2.5">
                  {history.map((item, idx) => (
                    <div
                      key={idx}
                      className="rounded-xl border border-purple-500/15 bg-[#080511] p-3 text-xs"
                    >
                      <div className="flex items-center justify-between font-mono">
                        <span className="uppercase text-purple-300">{item.action}</span>
                        <span className="font-bold text-white">{(item.prob * 100).toFixed(1)}% risk</span>
                      </div>
                      <p className="mt-1 truncate text-slate-400">{item.text}</p>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="space-y-2.5 font-mono text-xs text-slate-300">
                  {[
                    "1. Regex tokenizer extracts lowercase alphanumeric tokens",
                    "2. Log-normalized BoW vector mapped over 1,600-token vocab",
                    "3. Forward pass: Dense(96, ReLU) → Dense(32, ReLU) → Sigmoid(1)",
                    "4. Calibrated threshold gate routes to Allow / Review / Block",
                  ].map((step) => (
                    <div key={step} className="rounded-xl border border-purple-500/15 bg-[#080511] p-3">
                      {step}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </section>
      </section>
    </main>
  )
}
