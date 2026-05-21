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
  Sparkles,
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

const examples = [
  {
    label: "Safe review",
    text: "The delivery was late, but support helped me quickly and the product works well. I would buy it again.",
  },
  {
    label: "Spam prize",
    text: "WINNER! You have been selected for a free cash prize. Text CLAIM now to receive your reward.",
  },
  {
    label: "Urgent account scam",
    text: "URGENT: your account has been suspended. Click the secure link and verify your password immediately.",
  },
  {
    label: "Normal support",
    text: "Hi, can you tell me when my order will arrive? I placed it last week and need the tracking number.",
  },
]

function tokenize(text: string) {
  return text.toLowerCase().match(/[a-z0-9']+/g) || []
}

function relu(values: number[]) {
  return values.map((value) => Math.max(0, value))
}

function sigmoid(value: number) {
  return 1 / (1 + Math.exp(-value))
}

function dense(input: number[], kernel: number[][], bias: number[]) {
  return bias.map((biasValue, column) => {
    let sum = biasValue
    for (let row = 0; row < input.length; row += 1) {
      sum += input[row] * kernel[row][column]
    }
    return sum
  })
}

function vectorize(text: string, vocabulary: Record<string, number>) {
  const vector = new Array(Object.keys(vocabulary).length).fill(0)
  for (const token of tokenize(text)) {
    const index = vocabulary[token]
    if (index !== undefined) vector[index] += 1
  }
  return vector.map((value) => Math.log1p(value))
}

function predict(text: string, model: ModelExport) {
  const x = vectorize(text, model.vocabulary)
  const h1 = relu(dense(x, model.weights.dense1_kernel, model.weights.dense1_bias))
  const h2 = relu(dense(h1, model.weights.dense2_kernel, model.weights.dense2_bias))
  const logit = dense(h2, model.weights.out_kernel, model.weights.out_bias)[0]
  return sigmoid(logit)
}

function routing(probability: number, thresholds: ModelExport["thresholds"]) {
  if (probability >= thresholds.review) return "block"
  if (probability >= thresholds.allow) return "review"
  return "allow"
}

function actionCopy(action: string) {
  if (action === "block") return "High model confidence. Block the message and log a moderation event."
  if (action === "review") return "Borderline confidence. Send to a human moderator with model evidence."
  return "Low spam/abuse probability. Safe to publish."
}

export default function Home() {
  const [text, setText] = useState(examples[1].text)
  const [model, setModel] = useState<ModelExport | null>(null)
  const [loading, setLoading] = useState(true)
  const [runCount, setRunCount] = useState(1)

  useEffect(() => {
    fetch("/model/moderation_model.json")
      .then((response) => response.json())
      .then((data: ModelExport) => setModel(data))
      .finally(() => setLoading(false))
  }, [])

  const probability = useMemo(() => (model ? predict(text, model) : 0), [text, model, runCount])
  const action = model ? routing(probability, model.thresholds) : "loading"
  const percent = Math.round(probability * 1000) / 10
  const tokens = tokenize(text)
  const knownTokens = model ? tokens.filter((token) => model.vocabulary[token] !== undefined) : []
  const topTokens = Array.from(new Set(knownTokens)).slice(0, 12)

  return (
    <main className="min-h-screen bg-[#0b1018] text-slate-50">
      <section className="mx-auto grid min-h-screen max-w-7xl gap-8 px-6 py-10 lg:grid-cols-[390px_1fr]">
        <aside className="rounded-lg border border-white/10 bg-white/[0.04] p-5">
          <div className="mb-6 flex items-center gap-3">
            <Shield className="h-9 w-9 text-cyan-300" />
            <div>
              <h1 className="text-2xl font-black">Keras Moderation Model</h1>
              <p className="text-sm text-slate-400">Real trained neural spam/unsafe text classifier.</p>
            </div>
          </div>

          <div className="mb-4 flex flex-wrap gap-2">
            {examples.map((example) => (
              <button
                key={example.label}
                onClick={() => setText(example.text)}
                className="rounded-md border border-white/10 bg-slate-950 px-3 py-2 text-sm text-slate-300 transition hover:border-cyan-300/50 hover:text-cyan-100"
              >
                {example.label}
              </button>
            ))}
          </div>

          <textarea
            value={text}
            onChange={(event) => setText(event.target.value)}
            className="h-64 w-full resize-none rounded-md border border-white/10 bg-slate-950 p-4 text-sm leading-6 text-slate-100 outline-none transition focus:border-cyan-300/60"
            placeholder="Type any message. The trained Keras model runs inference in the browser..."
          />

          <button
            onClick={() => setRunCount((value) => value + 1)}
            disabled={!model}
            className="mt-4 flex w-full items-center justify-center gap-2 rounded-md bg-cyan-300 px-4 py-3 font-black text-slate-950 disabled:opacity-60"
          >
            {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
            Run Keras Inference
          </button>
        </aside>

        <section className="space-y-6">
          <div className="grid gap-4 md:grid-cols-4">
            {[
              ["Dataset", model?.metrics.dataset || "loading"],
              ["Test Accuracy", model ? `${(model.metrics.accuracy * 100).toFixed(1)}%` : "..."],
              ["F1 Score", model ? `${(model.metrics.f1 * 100).toFixed(1)}%` : "..."],
              ["Vocabulary", model ? String(model.metrics.vocabulary_size) : "..."],
            ].map(([label, value]) => (
              <div key={label} className="rounded-lg border border-white/10 bg-white/[0.04] p-4">
                <div className="text-sm text-slate-400">{label}</div>
                <div className="mt-2 text-2xl font-black text-cyan-200">{value}</div>
              </div>
            ))}
          </div>

          <div className="grid gap-6 lg:grid-cols-[1fr_360px]">
            <div className="rounded-lg border border-white/10 bg-white/[0.04] p-5">
              <div className="mb-4 flex items-center gap-2">
                <Brain className="h-5 w-5 text-cyan-300" />
                <h2 className="text-xl font-bold">Neural Model Prediction</h2>
              </div>
              <div className="rounded-lg border border-white/10 bg-slate-950 p-5">
                <div className="mb-2 flex items-center justify-between text-sm text-slate-400">
                  <span>Spam / unsafe probability</span>
                  <span>{percent}%</span>
                </div>
                <div className="h-4 rounded-full bg-white/10">
                  <div
                    className={`h-4 rounded-full ${action === "block" ? "bg-red-300" : action === "review" ? "bg-orange-300" : "bg-emerald-300"}`}
                    style={{ width: `${percent}%` }}
                  />
                </div>
                <div
                  className={`mt-5 inline-flex rounded-md px-3 py-2 text-sm font-black uppercase tracking-wide ${
                    action === "block"
                      ? "bg-red-300 text-slate-950"
                      : action === "review"
                        ? "bg-orange-300 text-slate-950"
                        : "bg-emerald-300 text-slate-950"
                  }`}
                >
                  {action}
                </div>
                <p className="mt-4 leading-7 text-slate-300">{actionCopy(action)}</p>
              </div>

              <div className="mt-5 rounded-lg border border-white/10 bg-slate-950 p-5">
                <div className="mb-3 flex items-center gap-2 text-sm font-bold text-slate-300">
                  <MessageSquareText className="h-4 w-4 text-emerald-300" />
                  Tested message
                </div>
                <p className="leading-8 text-slate-200">{text}</p>
              </div>
            </div>

            <div className="space-y-6">
              <div className="rounded-lg border border-white/10 bg-white/[0.04] p-5">
                <div className="mb-4 flex items-center gap-2">
                  <BarChart3 className="h-5 w-5 text-emerald-300" />
                  <h2 className="text-xl font-bold">Training Metrics</h2>
                </div>
                {model ? (
                  <div className="space-y-3 text-sm">
                    {[
                      ["Train samples", model.metrics.train_size],
                      ["Validation samples", model.metrics.validation_size],
                      ["Test samples", model.metrics.test_size],
                      ["Precision", `${(model.metrics.precision * 100).toFixed(1)}%`],
                      ["Recall", `${(model.metrics.recall * 100).toFixed(1)}%`],
                      ["Epochs", model.metrics.epochs_ran],
                    ].map(([label, value]) => (
                      <div key={label} className="flex justify-between rounded-md bg-slate-950 p-3">
                        <span className="text-slate-400">{label}</span>
                        <strong>{value}</strong>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-slate-400">Loading exported Keras model...</div>
                )}
              </div>

              <div className="rounded-lg border border-white/10 bg-[#f5f2ea] p-5 text-slate-950">
                <div className="mb-3 flex items-center gap-2 font-black">
                  <Database className="h-5 w-5" />
                  Real ML Pipeline
                </div>
                <p className="text-sm leading-6 text-slate-700">
                  The model was trained with Keras on the UCI SMS Spam Collection, exported as weights and vocabulary,
                  then executed directly in this web app.
                </p>
              </div>
            </div>
          </div>

          <div className="grid gap-6 lg:grid-cols-[1fr_360px]">
            <div className="rounded-lg border border-white/10 bg-white/[0.04] p-5">
              <div className="mb-4 flex items-center gap-2">
                <Sparkles className="h-5 w-5 text-cyan-300" />
                <h2 className="text-xl font-bold">Model Evidence</h2>
              </div>
              <div className="flex flex-wrap gap-2">
                {topTokens.length > 0 ? (
                  topTokens.map((token) => (
                    <span key={token} className="rounded-full border border-cyan-300/20 bg-cyan-300/10 px-3 py-1 text-sm text-cyan-100">
                      {token}
                    </span>
                  ))
                ) : (
                  <span className="text-slate-400">No vocabulary tokens matched.</span>
                )}
              </div>
            </div>

            <div className="rounded-lg border border-white/10 bg-white/[0.04] p-5">
              <div className="mb-4 flex items-center gap-2">
                <FileWarning className="h-5 w-5 text-orange-300" />
                <h2 className="text-xl font-bold">Audit Trace</h2>
              </div>
              <ol className="space-y-3 text-sm text-slate-300">
                {[
                  "Tokenize text with the same preprocessing used during training",
                  "Build log-scaled bag-of-words vector from 1,600-word vocabulary",
                  "Run Dense(96) -> Dense(32) -> sigmoid exported Keras weights",
                  "Apply allow/review/block thresholds",
                ].map((step, index) => (
                  <li key={step} className="flex gap-3">
                    <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-cyan-300 text-xs font-bold text-slate-950">
                      {index + 1}
                    </span>
                    <span>{step}</span>
                  </li>
                ))}
              </ol>
              <div className="mt-5 flex items-center gap-2 rounded-md border border-white/10 bg-white/[0.04] p-3 text-sm">
                {action === "allow" ? (
                  <>
                    <CheckCircle2 className="h-4 w-4 text-emerald-300" />
                    Safe to publish
                  </>
                ) : (
                  <>
                    <AlertOctagon className="h-4 w-4 text-red-300" />
                    Requires moderation action
                  </>
                )}
              </div>
            </div>
          </div>
        </section>
      </section>
    </main>
  )
}
