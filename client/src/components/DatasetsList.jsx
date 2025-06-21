import { useEffect, useState } from "react";
import { CalendarIcon, EyeIcon, TrashIcon } from "@heroicons/react/20/solid";
import { Link, useNavigate } from "react-router-dom";
import newlogo500 from "../assets/newlogo500.png";
import { ChatBubbleLeftEllipsisIcon } from "@heroicons/react/24/outline";

export default function DatasetsList() {
  const [datasets, setDatasets] = useState([]);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    const fetchDatasets = async () => {
      try {
        const res = await fetch("/api/datasets", {
          method: "GET",
          credentials: "include",
        });
        if (res.status === 401) {
          setError("Session expired. Please log in again.");
          navigate("/login");
          return;
        }
        if (!res.ok) {
          throw new Error("Failed to fetch datasets");
        }
        const data = await res.json();
        console.log("Fetched datasets:", data);
        setDatasets(data);
      } catch (err) {
        setError(err.message || "Unknown error");
      } finally {
        setLoading(false);
      }
    };
    fetchDatasets();
  }, [navigate]);

  const handleDelete = async (id) => {
    if (!window.confirm("Are you sure you want to delete this dataset?"))
      return;

    try {
      const res = await fetch(`/api/datasets/${id}`, {
        method: "DELETE",
        credentials: "include",
      });

      if (res.status === 204) {
        setDatasets((prev) => prev.filter((d) => d.id !== id));
      } else if (res.status === 401) {
        setError("Not authorized. Please log in again.");
        navigate("/login");
      } else {
        const msg = await res.text();
        throw new Error(msg || "Failed to delete dataset");
      }
    } catch (err) {
      setError(err.message);
    }
  };

  if (loading)
    return <div className="text-center py-8">Loading datasets...</div>;
  if (error)
    return <div className="text-red-500 text-center py-8">Error: {error}</div>;
  if (!datasets.length)
    return <div className="text-center py-8">No datasets found.</div>;

  return (
    <div className="bg-cyan-50 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <h2 className="text-2xl font-bold mb-2 text-blue-800">Saved Datasets</h2>
      <p className="text-blue-800 mt-1 mb-4 text-sm">
        Choose a dataset to begin analyzing and processing! File names with
        <img
          src={newlogo500}
          alt="Processed dataset"
          className="inline-block h-4 w-4 align-text-bottom mx-1"
        />
        have associated processed files saved. If you process and save them
        again, they will be overwritten.
      </p>

      <div className="flex justify-end mb-4">
        <Link
          to="/chat"
          className="inline-flex items-center space-x-1 bg-lime-500 hover:bg-cyan-700 text-white text-xs px-2 py-1 rounded"
        >
          <ChatBubbleLeftEllipsisIcon className="h-4 w-4" />
          <span>Chat with Databot!</span>
          <img src={newlogo500} alt="Data Tutor" className="h-4 w-4" />
        </Link>
      </div>

      <ul className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3">
        {datasets.map((dataset) => {
          const processed = dataset.has_cleaned_data;

          return (
            <li
              key={dataset.id}
              className="col-span-1 divide-y divide-gray-200 rounded-lg bg-white shadow"
            >
              <div className="flex flex-col space-y-3 p-6">
                <div className="flex-1">
                  <h3 className="text-lg font-semibold text-gray-900 truncate">
                    {dataset.title}
                    {processed && (
                      <img
                        src={newlogo500}
                        alt="Processed"
                        className="inline-block ml-2 h-4 w-4"
                      />
                    )}
                  </h3>
                  <p className="mt-1 text-sm text-gray-600 truncate">
                    {dataset.description}
                  </p>
                </div>
                <div className="flex items-center text-sm text-gray-500">
                  <CalendarIcon className="h-5 w-5 mr-1" aria-hidden="true" />
                  <span>
                    {new Date(dataset.uploaded_at).toLocaleDateString()}
                  </span>
                </div>
              </div>

              <div className="-mt-px flex divide-x divide-gray-200">
                <div className="flex w-0 flex-1">
                  <Link
                    to={`/datasets/${dataset.id}`}
                    className="relative -mr-px inline-flex w-0 flex-1 items-center justify-center gap-x-2 rounded-bl-lg bg-white px-2 py-2 text-xs font-semibold text-blue-900 hover:bg-indigo-50 hover:text-indigo-900"
                  >
                    <EyeIcon className="h-5 w-5" aria-hidden="true" />
                    View
                  </Link>
                </div>
                <div className="-ml-px flex w-0 flex-1">
                  <button
                    onClick={() => handleDelete(dataset.id)}
                    className="relative inline-flex w-0 flex-1 items-center justify-center gap-x-2 rounded-br-lg bg-blue-900 px-2 py-2 text-xs font-semibold text-white hover:bg-blue-700"
                  >
                    <TrashIcon className="h-5 w-5" aria-hidden="true" />
                    Delete
                  </button>
                </div>
              </div>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
