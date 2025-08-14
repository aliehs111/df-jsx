// client/src/components/ArchitectureDiagram.jsx
import archSvg from "../assets/dfjsx-arch.svg";

export default function ArchitectureDiagram({
  src = archSvg,
  caption = "df.jsx architecture overview",
}) {
  return (
    <figure className="mt-6">
      <img
        src={src}
        alt="System architecture diagram: React frontend, FastAPI backend, MySQL/JawsDB, and S3 storage."
        className="w-full h-auto rounded-lg border border-gray-200 shadow-sm"
        loading="lazy"
      />
      {caption ? (
        <figcaption className="mt-2 text-xs text-gray-500">
          {caption}
        </figcaption>
      ) : null}
    </figure>
  );
}
