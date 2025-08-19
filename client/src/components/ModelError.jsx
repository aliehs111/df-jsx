// src/components/ModelError.jsx
export default function ModelError({ message, onClose }) {
  if (!message) return null;

  return (
    <div className="mt-4 rounded-md border border-red-300 bg-red-50 p-3 text-sm text-red-800 flex items-start justify-between">
      <span>
        {message ||
          "This dataset or target may be unsuitable for the selected model."}
      </span>
      <button
        onClick={onClose}
        className="ml-3 text-red-700 hover:text-red-900 font-bold"
      >
        ×
      </button>
    </div>
  );
}
