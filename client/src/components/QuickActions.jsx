// src/components/QuickActions.jsx
import { Link } from "react-router-dom";

export default function QuickActions() {
  const actions = [
    {
      label: "Upload a CSV",
      to: "/upload",
      description: "Start by uploading a dataset to preview and clean.",
    },
    {
      label: "View all datasets",
      to: "/datasets",
      description: "Browse and manage all of your uploaded datasets.",
    },
    {
      label: "Run a model",
      to: "/models",
      description: "Select a dataset and run a quick model.",
    },
  ];

  return (
    <div>
      <h3 className="text-lg font-semibold text-blue-800">Quick actions</h3>
      <ul className="mt-4 space-y-3">
        {actions.map((action, idx) => (
          <li
            key={idx}
            className="flex items-center justify-between p-3 rounded-lg border border-gray-100 hover:border-blue-200 transition"
          >
            {/* Left side: text column */}
            <div className="flex flex-col text-left pr-4">
              <p className="text-sm font-medium text-blue-900">
                {action.label}
              </p>
              <p className="text-xs text-gray-500 mt-0.5">
                {action.description}
              </p>
            </div>

            {/* Right side: button */}
            <Link
              to={action.to}
              className="px-3 py-1.5 text-sm rounded border border-gray-200 hover:bg-gray-50"
            >
              Go
            </Link>
          </li>
        ))}
      </ul>
    </div>
  );
}
