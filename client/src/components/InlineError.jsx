// src/components/InlineError.jsx
export default function InlineError({ message, details, onClose }) {
  if (!message) return null;
  return (
    <div className="mt-4 rounded-md border border-red-300 bg-red-50 p-3 text-sm text-red-800 flex items-start justify-between">
      <div>
        <div>{message}</div>
        {details && (
          <details className="mt-1 text-red-700">
            <summary className="cursor-pointer underline">Details</summary>
            <pre className="mt-1 whitespace-pre-wrap text-xs">
              {typeof details === "string"
                ? details
                : JSON.stringify(details, null, 2)}
            </pre>
          </details>
        )}
      </div>
      <button
        onClick={onClose}
        className="ml-3 text-red-700 hover:text-red-900 font-bold"
        aria-label="Dismiss"
      >
        ×
      </button>
    </div>
  );
}
