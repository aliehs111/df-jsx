// components/Footer.jsx
export default function Footer() {
    return (
      <footer className="bg-gray-100 border-t py-4 mt-auto">
        <div className="container mx-auto flex justify-center items-center space-x-6">
          <img src="https://www.python.org/static/community_logos/python-logo.png" alt="Python" className="h-6" />
          <img src="https://pandas.pydata.org/static/img/pandas_white.svg" alt="Pandas" className="h-6" />
          <img src="https://numpy.org/images/logo.svg" alt="NumPy" className="h-6" />
          <img src="https://matplotlib.org/_static/images/logo2.svg" alt="Matplotlib" className="h-6" />
          <img src="https://seaborn.pydata.org/_static/logo-wide-lightbg.svg" alt="Seaborn" className="h-6" />
        </div>
      </footer>
    );
  }
  