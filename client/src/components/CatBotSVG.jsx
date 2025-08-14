// CatBotSVG.jsx — Databot brand mark (head + pendant)
// IDs are stable so your CSS (.db-blink, .db-tilt) can target them.

export default function CatBotSVG({
  size = 96, // change to 120 for larger
  stroke = "#233a88", // deep blue outline
  accent = "#16b9a6", // teal (eyes + pendant)
  fill = "#ffffff", // white face
  className = "",
  ...props
}) {
  return (
    <svg
      viewBox="0 0 160 160"
      width={size}
      height={size}
      aria-label="Databot logo"
      className={className}
      {...props}
    >
      <g
        id="db_logo"
        fill="none"
        stroke={stroke}
        strokeWidth="8"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        {/* Head + ears */}
        <g id="db_head">
          <path d="M51 26 L67 14 L68 36" />
          <path d="M109 26 L93 14 L92 36" />
          <rect x="36" y="28" width="88" height="76" rx="26" fill={fill} />
        </g>

        {/* Face bezel */}
        <g id="db_face">
          <rect x="50" y="40" width="60" height="48" rx="18" fill={fill} />
        </g>

        {/* Eyes (smile) — targeted by .db-blink */}
        <g id="db_eyes" stroke={accent} strokeWidth="8">
          <path d="M66 64 q8 10 16 0" />
          <path d="M94 64 q8 10 16 0" />
        </g>

        {/* Neck + pendant — pendant targeted by .db-tilt */}
        <g id="db_body">
          <path d="M80 104 v12" />
          <path d="M64 116 q16 20 32 0" />
          <circle
            id="db_pendant"
            cx="80"
            cy="124"
            r="12"
            fill={accent}
            stroke={stroke}
          />
        </g>
      </g>
    </svg>
  );
}
