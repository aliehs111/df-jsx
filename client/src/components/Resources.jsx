// src/components/Resources.jsx
import React from 'react';

export default function Resources() {
  const libs = [
    { name: 'Pandas',       url: 'https://pandas.pydata.org/docs/' },
    { name: 'NumPy',        url: 'https://numpy.org/doc/' },
    { name: 'scikit-learn', url: 'https://scikit-learn.org/stable/documentation.html' },
    { name: 'Matplotlib',   url: 'https://matplotlib.org/stable/contents.html' },
    { name: 'Seaborn',      url: 'https://seaborn.pydata.org/tutorial.html' },
    { name: 'SciPy',        url: 'https://docs.scipy.org/doc/scipy/reference/' },
  ];

  const datasets = [
    { name: 'Kaggle',         url: 'https://www.kaggle.com/datasets' },
    { name: 'UCI ML Repo',    url: 'https://archive.ics.uci.edu/ml/index.php' },
    { name: 'data.gov',       url: 'https://catalog.data.gov/' },
    { name: 'OpenML',         url: 'https://www.openml.org/search?type=data' },
    { name: 'Awesome Public Datasets',
                         url: 'https://github.com/awesomedata/awesome-public-datasets' },
  ];

  return (
    <div className="min-h-screen bg-cyan-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <h2 className="text-3xl font-bold mb-8 text-gray-800">Resources</h2>

        <section className="mb-12">
          <h3 className="text-2xl font-semibold mb-4 text-gray-700">Python Libraries</h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
            {libs.map((lib) => (
              <a
                key={lib.name}
                href={lib.url}
                target="_blank"
                rel="noopener noreferrer"
                className="block rounded-lg bg-white shadow p-6 hover:shadow-lg transition"
              >
                <h4 className="text-xl font-medium text-indigo-700">{lib.name}</h4>
                <p className="mt-2 text-sm text-gray-600">{lib.url}</p>
              </a>
            ))}
          </div>
        </section>

        <section>
          <h3 className="text-2xl font-semibold mb-4 text-gray-700">Public Dataset Sources</h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
            {datasets.map((src) => (
              <a
                key={src.name}
                href={src.url}
                target="_blank"
                rel="noopener noreferrer"
                className="block rounded-lg bg-white shadow p-6 hover:shadow-lg transition"
              >
                <h4 className="text-xl font-medium text-indigo-700">{src.name}</h4>
                <p className="mt-2 text-sm text-gray-600">{src.url}</p>
              </a>
            ))}
          </div>
        </section>
      </div>
    </div>
  );
}
