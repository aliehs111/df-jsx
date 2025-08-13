import React from "react";
import {
  SiPython,
  SiPandas,
  SiNumpy,
  SiScikitlearn,
  SiReact,
  SiTailwindcss,
  SiFastapi,
  SiDocker,
  SiGithub,
} from "react-icons/si";

// Inline SVGs for Matplotlib & Seaborn
function MatplotlibLogo({ className = "", color = "#11557c" }) {
  return (
    <svg
      viewBox="0 0 64 64"
      className={className}
      role="img"
      aria-label="Matplotlib"
    >
      <circle cx="32" cy="32" r="28" fill={color} opacity="0.15" />
      <path d="M32 32 L32 4 A28 28 0 0 1 60 32 Z" fill={color} opacity="0.35" />
      <path
        d="M32 32 L60 32 A28 28 0 0 1 32 60 Z"
        fill={color}
        opacity="0.55"
      />
      <circle cx="32" cy="32" r="4" fill={color} />
    </svg>
  );
}

function SeabornLogo({ className = "", color = "#4C72B0" }) {
  return (
    <svg
      viewBox="0 0 64 24"
      className={className}
      role="img"
      aria-label="Seaborn"
    >
      <g fill={color}>
        <circle cx="12" cy="12" r="10" opacity="0.45" />
        <circle cx="32" cy="12" r="10" opacity="0.75" />
        <circle cx="52" cy="12" r="10" opacity="0.45" />
      </g>
    </svg>
  );
}

// Navy base + brand hover colors
const items = [
  {
    name: "Python",
    icon: SiPython,
    href: "https://www.python.org/",
    color: "#3776AB",
  },
  {
    name: "Pandas",
    icon: SiPandas,
    href: "https://pandas.pydata.org/",
    color: "#150458",
  },
  {
    name: "NumPy",
    icon: SiNumpy,
    href: "https://numpy.org/",
    color: "#013243",
  },
  {
    name: "Matplotlib",
    svg: MatplotlibLogo,
    href: "https://matplotlib.org/",
    color: "#11557c",
  },
  {
    name: "Seaborn",
    svg: SeabornLogo,
    href: "https://seaborn.pydata.org/",
    color: "#4C72B0",
  },
  {
    name: "scikit-learn",
    icon: SiScikitlearn,
    href: "https://scikit-learn.org/",
    color: "#F7931E",
  },
  {
    name: "React",
    icon: SiReact,
    href: "https://react.dev/",
    color: "#61DAFB",
  },
  {
    name: "Tailwind CSS",
    icon: SiTailwindcss,
    href: "https://tailwindcss.com/",
    color: "#06B6D4",
  },
  {
    name: "FastAPI",
    icon: SiFastapi,
    href: "https://fastapi.tiangolo.com/",
    color: "#009688",
  },
  {
    name: "Docker",
    icon: SiDocker,
    href: "https://www.docker.com/",
    color: "#2496ED",
  },
  {
    name: "GitHub",
    icon: SiGithub,
    href: "https://github.com/",
    color: "#181717",
  },
];

export default function Footer() {
  return (
    <footer className="w-full border-t bg-neutralLight/80 backdrop-blur">
      <div className="mx-auto max-w-6xl px-4 py-3">
        <div className="flex flex-wrap items-center justify-center gap-x-6 gap-y-3">
          {items.map(({ name, icon: Icon, svg: Svg, href, color }) => (
            <a
              key={name}
              href={href}
              target="_blank"
              rel="noreferrer"
              aria-label={name}
              title={name}
              className="group inline-flex items-center"
            >
              {Icon ? (
                <Icon
                  className="h-6 w-6 transition-all duration-200"
                  style={{ color: "#1E3A8A" }} // navy base
                  onMouseEnter={(e) => (e.currentTarget.style.color = color)}
                  onMouseLeave={(e) =>
                    (e.currentTarget.style.color = "#1E3A8A")
                  }
                />
              ) : (
                <Svg
                  className="h-6 w-auto transition-all duration-200"
                  color="#1E3A8A"
                  onMouseEnter={(e) => {
                    e.currentTarget.querySelectorAll("*").forEach((el) => {
                      el.setAttribute("fill", color);
                    });
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.querySelectorAll("*").forEach((el) => {
                      el.setAttribute("fill", "#1E3A8A");
                    });
                  }}
                />
              )}
              <span className="sr-only">{name}</span>
            </a>
          ))}
        </div>
      </div>
    </footer>
  );
}
