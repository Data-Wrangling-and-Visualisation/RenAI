import React, { useState, useEffect } from "react";
import { Link } from "react-router-dom";

export default function HomePage({ artworksData }) {
  const [featuredArtwork, setFeaturedArtwork] = useState(null);

  useEffect(() => {
    if (artworksData && artworksData.items && artworksData.items.length > 0) {
      const randomIndex = Math.floor(Math.random() * artworksData.items.length);
      setFeaturedArtwork(artworksData.items[randomIndex]);
    }
  }, [artworksData]);

  const totalArtworks = artworksData?.items?.length || 0;
  const imageUrl = featuredArtwork?.primaryImageSmall || featuredArtwork?.cachedImageUrl || featuredArtwork?.primaryImage || 'https://source.unsplash.com/400x300/?abstract,art';

  return (
    <div className="min-h-screen bg-gray-900 text-white font-sans">
      <section
        className="relative h-screen flex items-center justify-center overflow-hidden"
        style={{ backgroundImage: "url('https://source.unsplash.com/1600x900/?art,technology,abstract')" }}
      >
        <div className="absolute inset-0 bg-black opacity-50 animate-pulse"></div>
        <div className="relative z-10 text-center space-y-6 px-4">
          <h1 className="text-5xl md:text-7xl font-extrabold tracking-wider drop-shadow-lg bg-gradient-to-r from-blue-400 via-purple-500 to-pink-500 bg-clip-text text-transparent animate-gradientShift">
            Art & AI Visions
          </h1>
          <p className="text-lg md:text-2xl max-w-2xl mx-auto animate-slideIn">
            Experience the powerful fusion of artistic creativity and advanced AI interpretation.
          </p>
          <Link
            to="/overview"
            className="inline-block px-8 py-4 bg-gradient-to-r from-purple-500 to-indigo-600 rounded-full font-semibold shadow hover:from-purple-600 hover:to-indigo-700 transform transition duration-300 hover:scale-110 animate-bounce"
          >
            Explore Now
          </Link>
        </div>
      </section>

      <section id="highlights" className="py-16 bg-gray-800">
        <div className="container mx-auto px-4 text-center">
          <h2 className="text-4xl font-bold mb-6 text-indigo-400">Collection</h2>
          <p className="text-xl mb-8 text-gray-300">
            Explore {totalArtworks} artworks from different collections.
          </p>
          {featuredArtwork && (
            <div className="max-w-md mx-auto bg-gray-700 rounded-lg shadow-lg overflow-hidden">
              <img 
                src={imageUrl} 
                alt={featuredArtwork.title || 'Featured Artwork'} 
                className="w-full h-64 object-cover"
                onError={(e) => { e.target.onerror = null; e.target.src='https://source.unsplash.com/400x300/?abstract,art'; }}
               />
              <div className="p-4">
                <h3 className="font-semibold text-lg mb-1">{featuredArtwork.title || 'Untitled'}</h3>
                <p className="text-sm text-gray-400 mb-2">{featuredArtwork.artistDisplayName || 'Unknown Artist'}</p>
                <Link
                   to={`/embeddings`}
                   onClick={() => console.log("Need to implement selection for featured artwork link")}
                   className="text-sm text-indigo-400 hover:text-indigo-300 transition duration-300"
                 >
                   Explore &rarr;
                 </Link>
              </div>
            </div>
          )}
        </div>
      </section>

      <section id="contact" className="py-20 bg-gray-900">
        <div className="container mx-auto px-4 text-center">
          <h2 className="text-4xl font-bold mb-6">Get in Touch</h2>
          <p className="text-lg mb-8">
            For collaborations, feedback, or inquiries, feel free to reach out.
          </p>
          <a
            href="mailto:d.ryazanov@innopolis.university"
            className="px-8 py-4 bg-indigo-600 rounded-full font-semibold shadow hover:bg-indigo-700 transform transition duration-300 hover:scale-105"
          >
            Contact Us
          </a>
        </div>
      </section>

      <footer className="bg-gray-800 py-6">
        <div className="container mx-auto text-center text-gray-500 text-sm">
          © {new Date().getFullYear()} Art & AI Visualizer. All Rights Reserved.
        </div>
      </footer>
    </div>
  );
}
