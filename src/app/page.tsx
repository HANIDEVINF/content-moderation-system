"use client"

import { useMemo, useState } from "react"
import { AlertOctagon, CheckCircle2, FileWarning, Lock, MessageSquareText, Radar, Shield, Workflow } from "lucide-react"

type PolicyKey = "toxicity" | "self_harm" | "privacy" | "fraud" | "violence" | "spam"
type Finding = {
  key: PolicyKey
  label: string
  score: number
  matches: string[]
  action: "allow" | "review" | "block"
}

const examples = [
  {
    label: "Clean Product Review",
    text: "The delivery was late, but support helped me quickly and the product works well. I would buy it again.",
  },
  {
    label: "PII Leak",
    text: "Here is my customer list: lina@example.com, +1 415 555 2199, and card 4111 1111 1111 1111. Please post it publicly.",
  },
  {
    label: "Harassment",
    text: "You are useless and everyone should spam this person until they disappear from the forum.",
  },
  {
    label: "Scam Campaign",
    text: "Click this urgent prize link and send your password so we can unlock your account bonus today.",
  },
]

const policies: Record<PolicyKey, { label: string; terms: string[]; weight: number; color: string }> = {
  toxicity: {
    label: "Toxicity",
    terms: ["useless", "idiot", "hate", "harass", "spam this person", "disappear"],
    weight: 28,
    color: "bg-red-300",
  },
  self_harm: {
    label: "Self Harm",
    terms: ["kill myself", "self harm", "end my life", "suicide"],
    weight: 35,
    color: "bg-purple-300",
  },
  privacy: {
    label: "Privacy",
    terms: ["password", "card", "customer list", "publicly"],
    weight: 30,
    color: "bg-cyan-300",
  },
  fraud: {
    label: "Fraud",
    terms: ["urgent prize", "unlock your account", "send your password", "bonus today"],
    weight: 32,
    color: "bg-orange-300",
  },
  violence: {
    label: "Violence",
    terms: ["attack", "weapon", "hurt them", "threat"],
    weight: 34,
    color: "bg-rose-300",
  },
  spam: {
    label: "Spam",
    terms: ["click this", "limited offer", "buy now", "free money"],
    weight: 22,
    color: "bg-lime-300",
  },
}

function extractRegexMatches(text: string) {
  const email = text.match(/\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b/g) || []
  const phone = text.match(/\+?\d[\d\s().-]{8,}\d/g) || []
  const card = text.match(/\b(?:\d[ -]*?){13,16}\b/g) || []
  return { email, phone, card }
}

function moderate(text: string): Finding[] {
  const lower = text.toLowerCase()
  const regexMatches = extractRegexMatches(text)

  return (Object.keys(policies) as PolicyKey[]).map((key) => {
    const policy = policies[key]
    const terms = policy.terms.filter((term) => lower.includes(term))
    const piiBoost = key === "privacy" ? regexMatches.email.length + regexMatches.phone.length + regexMatches.card.length : 0
    const rawScore = Math.min(99, terms.length * policy.weight + piiBoost * 24)
    const score = rawScore > 0 ? Math.max(38, rawScore) : 3
    const action = score >= 75 ? "block" : score >= 38 ? "review" : "allow"
    const matches = [...terms, ...(key === "privacy" ? [...regexMatches.email, ...regexMatches.phone, ...regexMatches.card] : [])]
    return { key, label: policy.label, score, matches, action }
  })
}

function overallAction(findings: Finding[]) {
  const max = Math.max(...findings.map((finding) => finding.score))
  if (max >= 75) return "block"
  if (max >= 38) return "review"
  return "allow"
}

export default function Home() {
  const [text, setText] = useState(examples[1].text)
  const findings = useMemo(() => moderate(text), [text])
  const action = overallAction(findings)
  const topFinding = [...findings].sort((a, b) => b.score - a.score)[0]
  const blocked = findings.filter((finding) => finding.action === "block").length
  const review = findings.filter((finding) => finding.action === "review").length

  return (
    <main className="min-h-screen bg-[#0b1018] text-slate-50">
      <section className="mx-auto grid min-h-screen max-w-7xl gap-8 px-6 py-10 lg:grid-cols-[380px_1fr]">
        <aside className="rounded-lg border border-white/10 bg-white/[0.04] p-5">
          <div className="mb-6 flex items-center gap-3">
            <Shield className="h-8 w-8 text-cyan-300" />
            <div>
              <h1 className="text-2xl font-bold">Content Moderation System</h1>
              <p className="text-sm text-slate-400">Policy scoring for user-generated text.</p>
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
            placeholder="Paste a user message, review, or chat transcript..."
          />

          <div className="mt-4 rounded-md border border-white/10 bg-slate-950 p-4">
            <div className="mb-2 flex items-center gap-2 text-sm text-slate-400">
              <Workflow className="h-4 w-4" />
              Routing decision
            </div>
            <div
              className={`inline-flex rounded-md px-3 py-2 text-sm font-bold uppercase tracking-wide ${
                action === "block"
                  ? "bg-red-300 text-slate-950"
                  : action === "review"
                    ? "bg-orange-300 text-slate-950"
                    : "bg-emerald-300 text-slate-950"
              }`}
            >
              {action}
            </div>
          </div>
        </aside>

        <section className="space-y-6">
          <div className="grid gap-4 md:grid-cols-4">
            {[
              ["Top Risk", topFinding.label],
              ["Risk Score", `${topFinding.score}%`],
              ["Blocks", String(blocked)],
              ["Reviews", String(review)],
            ].map(([label, value]) => (
              <div key={label} className="rounded-lg border border-white/10 bg-white/[0.04] p-4">
                <div className="text-sm text-slate-400">{label}</div>
                <div className="mt-2 text-2xl font-bold text-cyan-200">{value}</div>
              </div>
            ))}
          </div>

          <div className="rounded-lg border border-white/10 bg-white/[0.04] p-5">
            <div className="mb-4 flex items-center gap-2">
              <Radar className="h-5 w-5 text-cyan-300" />
              <h2 className="text-xl font-semibold">Policy Classifier</h2>
            </div>
            <div className="grid gap-4 md:grid-cols-2">
              {findings.map((finding) => (
                <div key={finding.key} className="rounded-md border border-white/10 bg-slate-950 p-4">
                  <div className="mb-3 flex items-center justify-between">
                    <div className="font-semibold">{finding.label}</div>
                    <div className="text-sm uppercase tracking-wide text-slate-400">{finding.action}</div>
                  </div>
                  <div className="h-3 rounded-full bg-white/10">
                    <div className={`h-3 rounded-full ${policies[finding.key].color}`} style={{ width: `${finding.score}%` }} />
                  </div>
                  <div className="mt-3 flex flex-wrap gap-2">
                    {finding.matches.length > 0 ? (
                      finding.matches.slice(0, 4).map((match) => (
                        <span key={match} className="rounded border border-white/10 bg-white/[0.04] px-2 py-1 text-xs text-slate-300">
                          {match}
                        </span>
                      ))
                    ) : (
                      <span className="text-xs text-slate-500">No matching signals</span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="grid gap-6 lg:grid-cols-[1fr_360px]">
            <div className="rounded-lg border border-white/10 bg-slate-950 p-5">
              <div className="mb-4 flex items-center gap-2">
                <MessageSquareText className="h-5 w-5 text-emerald-300" />
                <h2 className="text-xl font-semibold">Moderated Message</h2>
              </div>
              <p className="leading-8 text-slate-200">{text}</p>
            </div>

            <div className="rounded-lg border border-white/10 bg-slate-950 p-5">
              <div className="mb-4 flex items-center gap-2">
                <FileWarning className="h-5 w-5 text-orange-300" />
                <h2 className="text-xl font-semibold">Audit Trace</h2>
              </div>
              <ol className="space-y-3 text-sm text-slate-300">
                {[
                  "Normalize text and remove casing noise",
                  "Run regex detectors for email, phone, and payment patterns",
                  "Score policy categories with weighted lexical signals",
                  "Apply routing thresholds for allow, review, or block",
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
                ) : action === "review" ? (
                  <>
                    <Lock className="h-4 w-4 text-orange-300" />
                    Hold for moderator review
                  </>
                ) : (
                  <>
                    <AlertOctagon className="h-4 w-4 text-red-300" />
                    Block and log policy event
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
