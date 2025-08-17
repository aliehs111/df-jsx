// components/MetalButton.jsx
import { Link } from "react-router-dom";

export default function MetalButton({
  children,
  onClick,
  to,
  tone = "steel",
  className = "",
}) {
  const base = [
    "relative inline-flex items-center justify-center",
    "rounded-md px-4 py-2 text-sm font-semibold",
    "text-slate-800",
    "ring-1 ring-slate-300/80 hover:ring-slate-400 active:ring-slate-500",
    "transition-all duration-150 select-none",
    // subtle inner + outer shadow (gives metal edge)
    "shadow-[inset_0_1px_0_0_rgba(255,255,255,0.65),0_1px_2px_rgba(0,0,0,0.12)]",
    "active:translate-y-[0.5px] active:shadow-[inset_0_1px_0_0_rgba(255,255,255,0.6),0_0px_1px_rgba(0,0,0,0.18)]",
    "focus:outline-none focus-visible:ring-2 focus-visible:ring-slate-400/80",
    // glossy sheen overlay
    "before:absolute before:inset-0 before:rounded-md before:pointer-events-none",
    "before:bg-[linear-gradient(180deg,rgba(255,255,255,0.9)_0%,rgba(255,255,255,0.35)_38%,rgba(255,255,255,0.16)_50%,rgba(255,255,255,0.35)_64%,rgba(255,255,255,0.75)_100%)]",
    "before:opacity-70 hover:before:opacity-90",
    // crisp top highlight
    "after:absolute after:inset-x-0 after:top-0 after:h-px after:bg-white/70 after:rounded-t-md after:pointer-events-none",
  ].join(" ");

  const tones = {
    // brushed aluminum / silver
    steel:
      "bg-gradient-to-b from-zinc-50 via-zinc-200 to-zinc-300 hover:from-zinc-100 hover:to-zinc-400",
    // darker “titanium” option
    titanium:
      "bg-gradient-to-b from-slate-200 via-slate-300 to-slate-400 hover:to-slate-500",
    // soft gold accent (nice for Heatmap)
    gold: "bg-gradient-to-b from-amber-200 via-amber-300 to-amber-400 text-amber-900 ring-amber-300/70 hover:to-amber-500",
    // cool “blue steel” tint (nice for Insights)
    blueSteel:
      "bg-gradient-to-b from-sky-100 via-sky-200 to-sky-300 text-slate-900 ring-sky-300/70 hover:to-sky-400",
  };

  const cls = [base, tones[tone], className].join(" ");
  if (to)
    return (
      <Link to={to} className={cls}>
        {children}
      </Link>
    );
  return (
    <button onClick={onClick} className={cls}>
      {children}
    </button>
  );
}
