import React, { useState, useEffect, useCallback, useRef } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import {
  Box, Alert, Dialog, DialogActions, DialogContent, DialogContentText, 
  DialogTitle, Button, CircularProgress, Typography, Snackbar, IconButton 
} from '@mui/material';
import {
  FileUpload as FileUploadIcon,
  Close as CloseIcon
} from '@mui/icons-material';
import Sidebar from './components/Sidebar';
import EmbeddingProjection from './components/EmbeddingProjection';
import AttentionVisualizer from './components/AttentionVisualizer';
import ArtSimilarityGraph from './components/ArtSimilarityGraph';
import SimilarityGallery from './components/SimilarityHeatmap';
import StyleAnalysis from './components/StyleAnalysis';
import GlobalImagePanel from './components/GlobalImagePanel';
import HomePage from './pages/HomePage';
import OverviewPage from './components/OverviewPage';

// Initialize IndexedDB
const initIndexedDB = () => {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open('ArtworkCache', 1);
    
    request.onerror = event => {
      console.error('IndexedDB error:', event.target.error);
      reject(event.target.error);
    };
    
    request.onupgradeneeded = event => {
      const db = event.target.result;
      if (!db.objectStoreNames.contains('images')) {
        db.createObjectStore('images', { keyPath: 'id' });
      }
      if (!db.objectStoreNames.contains('session')) {
        db.createObjectStore('session', { keyPath: 'id' });
      }
    };
    
    request.onsuccess = event => {
      resolve(event.target.result);
    };
  });
};

// Save image to IndexedDB
const saveImageToIndexedDB = async (objectId, imageData) => {
  // --- DISABLE SAVING --- 
  console.log(`saveImageToIndexedDB called for ${objectId}, but saving is disabled.`);
  return Promise.resolve(false); // Immediately return false, do nothing
  // --- END DISABLE ---
  /* --- Original Code Commented Out ---
  try {
    const db = await initIndexedDB();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction(['images'], 'readwrite');
      const store = transaction.objectStore('images');
      
      const request = store.put({ id: `artwork_image_${objectId}`, data: imageData, timestamp: Date.now() });
      
      request.onsuccess = () => resolve(true);
      request.onerror = event => {
        console.error('Error saving image to IndexedDB:', event.target.error);
        reject(event.target.error);
      };
    });
  } catch (error) {
    console.error('Failed to save image to IndexedDB:', error);
    return false;
  }
  // --- END DISABLE --- */
};

// Get image from IndexedDB
const getImageFromIndexedDB = async (objectId) => {
  // --- DISABLE READING --- 
  console.log(`getImageFromIndexedDB called for ${objectId}, but reading is disabled.`);
  return Promise.resolve(null); // Immediately return null
  // --- END DISABLE ---
  /* --- Original Code Commented Out ---
  try {
    const db = await initIndexedDB();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction(['images'], 'readonly');
      const store = transaction.objectStore('images');
      
      const request = store.get(`artwork_image_${objectId}`);
      
      request.onsuccess = event => {
        if (event.target.result) {
          resolve(event.target.result.data);
        } else {
          resolve(null);
        }
      };
      
      request.onerror = event => {
        console.error('Error getting image from IndexedDB:', event.target.error);
        reject(event.target.error);
      };
    });
  } catch (error) {
    console.error('Failed to get image from IndexedDB:', error);
    return null;
  }
  // --- END DISABLE --- */
};

// Save session data
const saveSessionData = async (data) => {
  try {
    const db = await initIndexedDB();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction(['session'], 'readwrite');
      const store = transaction.objectStore('session');
      
      const request = store.put({ id: 'currentSession', ...data, timestamp: Date.now() });
      
      request.onsuccess = () => resolve(true);
      request.onerror = event => {
        console.error('Error saving session to IndexedDB:', event.target.error);
        reject(event.target.error);
      };
    });
  } catch (error) {
    console.error('Failed to save session to IndexedDB:', error);
    return false;
  }
};

// Get session data
const getSessionData = async () => {
  try {
    const db = await initIndexedDB();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction(['session'], 'readonly');
      const store = transaction.objectStore('session');
      
      const request = store.get('currentSession');
      
      request.onsuccess = event => {
        if (event.target.result) {
          resolve(event.target.result);
        } else {
          resolve(null);
        }
      };
      
      request.onerror = event => {
        console.error('Error getting session from IndexedDB:', event.target.error);
        reject(event.target.error);
      };
    });
  } catch (error) {
    console.error('Failed to get session from IndexedDB:', error);
    return null;
  }
};

const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#3f51b5',
    },
    secondary: {
      main: '#f50057',
    },
    background: {
      default: '#f5f5f7',
      paper: '#ffffff'
    }
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    h5: {
      fontWeight: 500,
    },
    h6: {
      fontWeight: 500,
    }
  }
});

const App = () => {
  const [artworksData, setArtworksData] = useState({
      items: [], 
      embeddings: [],
  }); 
  const [currentPage, setCurrentPage] = useState(1);
  const [totalArtworks, setTotalArtworks] = useState(0);
  const [isLoadingInitial, setIsLoadingInitial] = useState(true);
  const [isLoadingMore, setIsLoadingMore] = useState(false);
  const [errorLoading, setErrorLoading] = useState(null);
  const [sessionLoaded, setSessionLoaded] = useState(false);

  const [selectedArtworkData, setSelectedArtworkData] = useState(null);
  const [attentionMapUrl, setAttentionMapUrl] = useState(null);
  const [gradcamMapUrl, setGradcamMapUrl] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [errorProcessing, setErrorProcessing] = useState(null);
  const [showUploadDialog, setShowUploadDialog] = useState(false);

  // --- ADD State for Upload Dialog --- 
  const [uploadedImage, setUploadedImage] = useState(null); // Stores the preview data URL
  const [uploadedImageFile, setUploadedImageFile] = useState(null); // Stores the actual File object
  const [uploadError, setUploadError] = useState(null);
  const [uploadLoading, setUploadLoading] = useState(false);
  const [dragActive, setDragActive] = useState(false); // For drag-n-drop visual feedback
  const [snackbarOpen, setSnackbarOpen] = useState(false); // For success message
  const [snackbarMessage, setSnackbarMessage] = useState('');
  const fileInputRef = useRef(null); // For triggering file input click
  // --- END ADD State ---

  const BACKEND_URL = 'http://localhost:5000';
  const ARTWORKS_PER_PAGE = 50;
  // const INITIAL_RANDOM_ARTWORKS = 15; // This seems unused now

  // --- MOVED UP: Save Session Function --- 
  const saveSession = useCallback(async () => {
    // Only save if session has been loaded initially to avoid overwriting
    if (!sessionLoaded) {
      console.log("Session not loaded yet, skipping save.");
      return;
    }
    console.log("Attempting to save session...");
    const sessionData = {
      artworksData: {
          items: artworksData.items.slice(0, 100), // Example: Save only first 100 items
          embeddings: artworksData.embeddings.slice(0, 100) // Save corresponding embeddings
      },
      currentPage,
      totalArtworks,
      selectedArtworkData: selectedArtworkData ? { ...selectedArtworkData } : null,
      attentionMapUrl,    
      gradcamMapUrl       
    };
    try {
      // Ensure saveSessionData (from utils) is defined/imported correctly
      await saveSessionData(sessionData); 
      console.log('Session saved successfully');
    } catch (error) {
      console.error('Failed to save session:', error);
    }
  }, [
    sessionLoaded, artworksData, currentPage, totalArtworks, 
    selectedArtworkData, attentionMapUrl, gradcamMapUrl
    // saveSessionData is defined outside the component, so it's stable
  ]);
  // --- END MOVED UP ---

  // --- Load Session Function ---
  const loadSession = useCallback(async () => {
    try {
      // Try to load session from IndexedDB
      const session = await getSessionData();
      
      if (session && session.artworksData && session.artworksData.items.length > 0) {
        console.log('Restoring session from IndexedDB');
        
        // --- ADD FILTER: Remove user uploads from restored session data --- 
        const filteredItems = session.artworksData.items.filter(item => item?.department !== 'User Uploads' && item?.type !== 'uploaded');
        const filteredEmbeddings = session.artworksData.embeddings.filter((emb, index) => {
            const item = session.artworksData.items[index]; // Get corresponding item
            return item?.department !== 'User Uploads' && item?.type !== 'uploaded';
        });
        console.log(`Restored session, filtered out ${session.artworksData.items.length - filteredItems.length} user uploads.`);
        // --- END FILTER ---

        // --- Use filtered data ---
        setArtworksData({
            items: filteredItems,
            embeddings: filteredEmbeddings
        });
        // --- End Use filtered data ---

        setCurrentPage(session.currentPage || 1);
        setTotalArtworks(session.totalArtworks || 0);
        
        if (session.selectedArtworkData) {
          // --- Check if selected artwork is uploaded, if so, don't restore it --- 
          if (session.selectedArtworkData?.metadata?.type !== 'uploaded') {
              setSelectedArtworkData(session.selectedArtworkData);
              
              const objectID = session.selectedArtworkData.metadata?.objectID || session.selectedArtworkData.objectID;
              if (objectID) {
                  setGradcamMapUrl(session.gradcamMapUrl || `${BACKEND_URL}/api/gradcam_image/${objectID}`);
                  setAttentionMapUrl(session.attentionMapUrl || `${BACKEND_URL}/api/attention_image/${objectID}`);
              }
          } else {
              console.log("Skipping restoration of selected artwork as it was a user upload.");
          }
          // --- End Check ---
        }
        
        //setIsLoadingInitial(false); // Moved this setting to the initializeApp effect
        //setSessionLoaded(true);
        
        return true; // Indicate session was restored
      } else {
           console.log("No valid session found in IndexedDB.");
      }
      
      return false;
    } catch (error) {
      console.error('Error loading session:', error);
      return false;
    }
  }, []);

  // --- Image Caching/Fetching Utilities ---
  const blobToBase64 = (blob) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => resolve(reader.result);
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });
  };

  const fetchAndCacheImage = useCallback(async (imageUrl, objectId) => {
    console.log(`%%% fetchAndCacheImage START for ID: ${objectId} %%%`); // <-- ADDED ENTRY LOG
    console.log(`[fetchAndCacheImage] ID: ${objectId}, Input URL: ${imageUrl ? imageUrl.substring(0,60)+'...' : 'null'}`);
    if (!imageUrl) {
        console.warn(`[fetchAndCacheImage] No URL for ${objectId}`);
        return null;
    }

    try {
        if (imageUrl.startsWith('http')) {
            const proxyUrl = `${BACKEND_URL}/api/proxy_image?url=${encodeURIComponent(imageUrl)}`;
            try {
                console.log(`[fetchAndCacheImage] Fetching via proxy: ${proxyUrl.substring(0, 100)}...`);
                console.time(`[fetchAndCacheImage] Proxy fetch for ${objectId}`);
                const response = await fetch(proxyUrl);
                console.timeEnd(`[fetchAndCacheImage] Proxy fetch for ${objectId}`);
                
                console.log(`[fetchAndCacheImage] Proxy response status for ${objectId}: ${response.status}, Content-Type: ${response.headers.get('content-type')}`);
                
                if (!response.ok) {
                    throw new Error(`Proxy fetch failed for ${objectId}: ${response.status} ${response.statusText}`);
                }
                
                console.time(`[fetchAndCacheImage] Blob extraction for ${objectId}`);
                const blob = await response.blob();
                console.timeEnd(`[fetchAndCacheImage] Blob extraction for ${objectId}`);
                
                console.log(`[fetchAndCacheImage] Blob received for ${objectId}: size=${blob.size}, type=${blob.type}`);
                
                if (blob.size === 0) {
                     throw new Error(`Proxy fetch for ${objectId} returned empty blob.`);
                }
                
                console.time(`[fetchAndCacheImage] Base64 conversion for ${objectId}`);
                const base64data = await blobToBase64(blob);
                console.timeEnd(`[fetchAndCacheImage] Base64 conversion for ${objectId}`);
                
                console.log(`[fetchAndCacheImage] Proxy success for ${objectId}, returning base64 length: ${base64data ? base64data.length : 0}`);
                return base64data;
            } catch (error) {
                 console.error(`[fetchAndCacheImage] Proxy/Fetch Error for ${objectId}:`, error);
                 return null; // Explicitly return null on error
            }
        } else if (imageUrl.startsWith('data:image/')) {
             console.log(`[fetchAndCacheImage] URL for ${objectId} is already base64, length: ${imageUrl.length}`);
             return imageUrl; // Return as is for base64 data
        } else {
             console.warn(`[fetchAndCacheImage] Unsupported URL format for ${objectId}: ${imageUrl.substring(0, 20)}...`);
             return null; // Return null for unsupported formats
        }
    } catch (error) {
        console.error(`[fetchAndCacheImage] Unexpected error for ${objectId}:`, error);
        return null;
    } finally {
        console.log(`%%% fetchAndCacheImage END for ID: ${objectId} %%%`);
    }
}, [BACKEND_URL]);

  // --- Fetch Initial Artworks Data --- 
  const fetchArtworksData = useCallback(async () => {
    console.log("%%% fetchArtworksData START %%%"); 
    console.log("App: Fetching initial artworks data using /api/museum_artworks with fetch..."); // <-- Updated log
    try {
      // Using native fetch instead of axios
      console.log("%%% fetchArtworksData: BEFORE fetch call"); // <-- Log before fetch
      const response = await fetch(`${BACKEND_URL}/api/museum_artworks?limit=${ARTWORKS_PER_PAGE}`);
      console.log("%%% fetchArtworksData: AFTER fetch call"); // <-- Log after fetch

      console.log("%%% fetchArtworksData: API Response Status:", response.status);
      
      if (!response.ok) {
          // Attempt to get error message from response body
          let errorMsg = `HTTP error! status: ${response.status}`;
          try {
              const errorData = await response.json();
              errorMsg = errorData.error || errorMsg;
          } catch (e) {
              // Ignore if response body is not JSON or empty
              console.warn("Could not parse error response body");
          }
          throw new Error(errorMsg);
      }

      const responseData = await response.json(); // Parse JSON data
      console.log("%%% fetchArtworksData: API Response Data (keys):", responseData ? Object.keys(responseData) : 'null');
      console.log("%%% fetchArtworksData: API Response Data Sample:", 
                  responseData?.metadata?.slice(0, 1), 
                  "Embeddings length:", responseData?.embeddings?.length);
      
      const museumData = responseData || { metadata: [], embeddings: [] };
      const museumItems = museumData.metadata || [];
      const museumEmbeddings = museumData.embeddings || [];
      
      if (museumItems.length !== museumEmbeddings.length) {
        console.warn(`App: Length mismatch between metadata and embeddings: metadata=${museumItems.length}, embeddings=${museumEmbeddings.length}`);
      }

      // --- Process, Filter, and Pre-fetch Images ---
      const finalUniqueItems = [];
      const finalUniqueEmbeddings = [];
      const seenIds = new Set();
      const minLength = Math.min(museumItems.length, museumEmbeddings.length);
      console.log(`App: Processing ${minLength} entries from /api/museum_artworks`);

      // Create promises for fetching/caching images concurrently
      const imageFetchPromises = [];

      for (let i = 0; i < minLength; i++) {
          const item = museumItems[i];
          const embedding = museumEmbeddings[i];
          const itemId = item?.id || item?.objectID;

          if (item?.department === 'User Uploads' || item?.type === 'uploaded') continue;

          if (item && itemId && embedding && !seenIds.has(itemId)) {
              item.id = itemId; // Ensure consistent ID
              console.log(`%%% Processing item ${i}: ID=${itemId}, Title=${item.title}, Has primaryImageSmall=${!!item.primaryImageSmall}, Has primaryImage=${!!item.primaryImage}`);

              // *** Start fetching/caching the image ***
              const imageUrl = item.primaryImageSmall || item.primaryImage;
              if (imageUrl) {
                 console.log(`%%% Starting image fetch for item ${i}: ID=${itemId}, URL start: ${imageUrl.substring(0, 30)}...`);
                 // Add the promise to the array, store the item index
                 imageFetchPromises.push(
                     fetchAndCacheImage(imageUrl, itemId).then(cachedUrl => ({ index: i, cachedUrl }))
                 );
              } else {
                 console.warn(`%%% Item ${i} with ID=${itemId} has no image URL`);
                 item.cachedImageUrl = null; // Ensure property exists even if no URL
              }
              // *** End image fetch initiation ***

              finalUniqueItems.push(item); // Add item metadata immediately
              finalUniqueEmbeddings.push(embedding); // Add embedding immediately
              seenIds.add(itemId);
          } else {
               console.warn(`App: Skipping item ${i}: ID=${itemId}, HasItem=${!!item}, HasEmbedding=${!!embedding}, AlreadySeen=${seenIds.has(itemId)}`);
          }
      }

      // --- Wait for all image fetches to complete --- 
      console.log(`App: Waiting for ${imageFetchPromises.length} image fetch/cache promises...`);
      const imageResults = await Promise.allSettled(imageFetchPromises);
      console.log("App: Image fetch/cache promises settled, successful:", imageResults.filter(r => r.status === 'fulfilled').length, 
                  "failed:", imageResults.filter(r => r.status === 'rejected').length);

      // --- Update items with cached URLs --- 
      imageResults.forEach(result => {
          if (result.status === 'fulfilled') {
              const { index, cachedUrl } = result.value || {};
              const originalItem = museumItems[index]; 
              const itemId = originalItem?.id || originalItem?.objectID;
              const itemIndexInFinal = finalUniqueItems.findIndex(finalItem => finalItem.id === itemId);
              if (itemIndexInFinal !== -1) {
                 finalUniqueItems[itemIndexInFinal].cachedImageUrl = cachedUrl;
                 // *** ADD LOGGING HERE ***
                 console.log(`[fetchArtworksData] Assigned cachedUrl to ID ${itemId}: ${cachedUrl ? cachedUrl.substring(0,60)+'...' : 'null'}`);
              } else {
                 console.warn(`[fetchArtworksData] Could not find final item for ID ${itemId} to assign cached URL.`);
              }
          } else if (result.status === 'rejected') {
              console.error("[fetchArtworksData] An image fetch promise failed:", result.reason);
          }
      });
      
      console.log("[fetchArtworksData] Final items before setting state:", finalUniqueItems.slice(0, 5)); // Log first 5 items
      console.log("[fetchArtworksData] Final embeddings before setting state (first 5):", finalUniqueEmbeddings.slice(0, 5).map(e => e?.slice(0, 10))); 
      
      console.log(`App: Final unique data after image processing - Items: ${finalUniqueItems.length}, Embeddings: ${finalUniqueEmbeddings.length}`);
      
      if (finalUniqueItems.length === 0) {
           console.warn("App: No valid artworks loaded from /api/museum_artworks.");
           setArtworksData({ items: [], embeddings: [] }); 
      } else {
           setArtworksData({
              items: finalUniqueItems, // Now contains items with cachedImageUrl (hopefully)
              embeddings: finalUniqueEmbeddings,
           });
           setTotalArtworks(museumData.total || finalUniqueItems.length);
           setCurrentPage(museumData.page || 1);
           console.log("App: Successfully updated artworksData from /api/museum_artworks including image pre-fetch.");
      }
      
    } catch (error) {
      console.error('%%% fetchArtworksData: CAUGHT ERROR:', error);
      console.error('App: Error fetching initial artworks data:', error);
      setErrorLoading(error.message || 'Failed to load initial artwork data.');
      setArtworksData({ items: [], embeddings: [] }); // Reset on error
    } 
  }, [BACKEND_URL, ARTWORKS_PER_PAGE, fetchAndCacheImage]); // Added fetchAndCacheImage dependency

  // --- Fetch Analysis AND Image for Selected Artwork --- 
  const fetchNewAnalysisAndImage = useCallback(async (newId) => {
    if (!newId) {
      console.error("fetchNewAnalysisAndImage: Called with invalid ID:", newId);
      return { metadata: null, analysis: null }; // Return empty object on invalid ID
    }

    setIsProcessing(true);
    setErrorProcessing(null);
    console.log(`fetchNewAnalysisAndImage: Fetching analysis and image for ID: ${newId}`);

    try {
      // Find the base artwork metadata locally first
      const baseArtworkIndex = artworksData.items.findIndex(item => (item.id || item.objectID) === newId);
      let baseArtwork = baseArtworkIndex !== -1 ? { ...artworksData.items[baseArtworkIndex] } : null; // Create a copy

      if (!baseArtwork) {
        console.error(`fetchNewAnalysisAndImage: Artwork metadata not found locally for ID ${newId}`);
        setErrorProcessing(`Metadata not found for artwork ${newId}`);
        return { metadata: null, analysis: null }; // Return empty
      }

      // --- Concurrently fetch analysis and image (if needed) --- 
      const analysisFetchPromise = (async () => {
           let styleResult = null, colorResult = null, compositionResult = null;
           try {
               const [styleResponse, colorResponse, compositionResponse] = await Promise.all([
                   fetch(`${BACKEND_URL}/api/analyze/style/${newId}`),
                   fetch(`${BACKEND_URL}/api/analyze/color/${newId}`),
                   fetch(`${BACKEND_URL}/api/analyze/composition/${newId}`)
               ]);
               if (styleResponse.ok) styleResult = await styleResponse.json();
               if (colorResponse.ok) colorResult = await colorResponse.json();
               if (compositionResponse.ok) compositionResult = await compositionResponse.json();
               return { style: styleResult, color: colorResult, composition: compositionResult };
           } catch (err) {
               console.error(`Error fetching analysis bundle for ${newId}:`, err);
               // Set analysis error state? For now, return nulls
               setErrorProcessing(`Analysis fetch failed: ${err.message}`); 
               return { style: null, color: null, composition: null }; 
           }
      })();

      const imageFetchPromise = (async () => {
          // Only fetch if cachedImageUrl is missing or not base64
          if (!baseArtwork.cachedImageUrl || !baseArtwork.cachedImageUrl.startsWith('data:image')) {
              const imageUrl = baseArtwork.primaryImageSmall || baseArtwork.primaryImage;
              if (imageUrl) {
                  console.log(`fetchNewAnalysisAndImage: Fetching/caching image for selected ID ${newId}`);
                  return await fetchAndCacheImage(imageUrl, newId);
              }
          }
          return baseArtwork.cachedImageUrl; // Return existing URL if valid
      })();
      // --- End Concurrent Fetches --- 

      // --- Wait for both and combine --- 
      const [analysisResult, cachedImageUrlResult] = await Promise.all([
          analysisFetchPromise,
          imageFetchPromise
      ]);

      // Update the base artwork copy with the fetched/cached image URL
      baseArtwork.cachedImageUrl = cachedImageUrlResult;
      console.log(`fetchNewAnalysisAndImage: Final cachedImageUrl for ${newId}:`, cachedImageUrlResult ? cachedImageUrlResult.substring(0,60)+'...' : null);

      // --- Construct the final analysis object INCLUDING map URLs ---
      const finalAnalysis = {
          ...(analysisResult || {}), // Spread existing style, color, composition
          attention: {
              ...(analysisResult?.attention || {}), // Keep existing attention data if any
              map_url: `${BACKEND_URL}/api/attention_image/${newId}?t=${Date.now()}` // Add/overwrite map_url
          },
          gradcam: {
              ...(analysisResult?.gradcam || {}), // Keep existing gradcam data if any
              map_url: `${BACKEND_URL}/api/gradcam_image/${newId}?t=${Date.now()}` // Add/overwrite map_url
          }
      };
      // --- End construct final analysis object --- 

      // --- Update main artworksData state with the new cached URL for this item --- 
      // This ensures the list updates if the image was just cached
      setArtworksData(prevData => ({
           ...prevData,
           items: prevData.items.map((item, index) => 
               (item.id || item.objectID) === newId 
                 ? { ...item, cachedImageUrl: cachedImageUrlResult } // Update the specific item
                 : item
           )
      }));
      
      // Return combined data for setting selectedArtworkData
      return { metadata: baseArtwork, analysis: finalAnalysis };

    } catch (error) {
      console.error(`fetchNewAnalysisAndImage: Error processing ID ${newId}:`, error);
      setErrorProcessing(`Failed to load data: ${error}`);
      return { metadata: baseArtwork, analysis: null }; // Return metadata even if analysis fails
    } finally {
      setIsProcessing(false);
      console.log(`fetchNewAnalysisAndImage: Finished processing for ID: ${newId}`);
    }
  }, [artworksData.items, BACKEND_URL, fetchAndCacheImage]); // Dependencies

  // --- Handle Artwork Selection --- 
  const handleArtworkSelect = useCallback(async (objectId) => {
    if (!objectId) return;
    
    // Call the new function to fetch analysis and image
    const { metadata, analysis } = await fetchNewAnalysisAndImage(objectId); 

    // Update the selected artwork state
    if (metadata) { // Only update if metadata was found/returned
      const embedding = artworksData.embeddings[artworksData.items.findIndex(item => (item.id || item.objectID) === objectId)] || null;
      setSelectedArtworkData({ metadata, analysis, embedding });
      console.log("handleArtworkSelect: Updated selectedArtworkData for ID:", objectId);
      // Update attention/gradcam maps (consider moving this into fetchNewAnalysisAndImage if always needed)
      setGradcamMapUrl(`${BACKEND_URL}/api/gradcam_image/${objectId}?t=${Date.now()}`);
      setAttentionMapUrl(`${BACKEND_URL}/api/attention_image/${objectId}?t=${Date.now()}`);
    } else {
        // If metadata came back null (error handled in fetch func), maybe clear selection
    setSelectedArtworkData(null);
    setGradcamMapUrl(null);
    setAttentionMapUrl(null);
    }

  }, [fetchNewAnalysisAndImage, artworksData.items, artworksData.embeddings, BACKEND_URL]); // Dependencies

  // --- Load Initial Data and Session (useEffect) ---
  useEffect(() => {
    const initializeApp = async () => {
      console.log("Initializing App...");
      setIsLoadingInitial(true);

      try {
        const sessionRestored = await loadSession(); // Depends on stable loadSession
        console.log("%%% sessionRestored value: ", sessionRestored); // <-- ADDED LOG HERE
        // if (!sessionRestored) { // <-- TEMPORARILY COMMENTED OUT for debugging
            console.log("No session restored OR FORCING FETCH, fetching initial data..."); // <-- Adjusted log
            await fetchArtworksData(); // Call fetchArtworksData directly here
        // } else { // <-- TEMPORARILY COMMENTED OUT
            // console.log("Session restored."); // <-- TEMPORARILY COMMENTED OUT
        // }
      } catch (error) {
        console.error("Error during initialization:", error);
        setErrorLoading("Initialization failed. Trying to fetch data directly.");
        // Fallback fetch might be needed here depending on desired behavior
        // await fetchArtworksData(); 
      } finally {
        setIsLoadingInitial(false);
        setSessionLoaded(true);
      }
    };

    initializeApp();

  // *** CORRECTED DEPENDENCIES ***
  // Only include stable functions needed for the one-time initialization logic.
  // fetchArtworksData and handleArtworkSelect should NOT be here.
  }, []); // <-- CHANGED to empty dependency array for mount-only execution

  // --- Auto-select first artwork AFTER data is loaded/restored --- 
  useEffect(() => {
    if (!isLoadingInitial && !selectedArtworkData && artworksData.items.length > 0 && typeof handleArtworkSelect === 'function') {
      const timer = setTimeout(() => {
           console.log("Running auto-selection effect.");
           handleArtworkSelect(artworksData.items[0].id);
      }, 0); 
      return () => clearTimeout(timer); 
    }
  // Dependencies are correct now
  }, [isLoadingInitial, artworksData.items, selectedArtworkData, handleArtworkSelect]); 

  // --- Save current session whenever important state changes --- 
  useEffect(() => {
    // Debounce or throttle saveSession if it gets called too frequently
    if (sessionLoaded) {
        const timer = setTimeout(() => {
             saveSession();
        }, 1000); // Save session 1 second after changes settle
        return () => clearTimeout(timer); // Cleanup timer
    }
    // Dependencies include all state that should trigger a save + saveSession itself
  }, [sessionLoaded, artworksData, currentPage, totalArtworks, selectedArtworkData, attentionMapUrl, gradcamMapUrl, saveSession]);

  // --- Upload Dialog Handlers & Analysis Logic ---
  const handleClearUploadedImage = useCallback(() => {
    setUploadedImage(null);
    setUploadedImageFile(null);
    setUploadError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = ''; // Reset file input
    }
  }, []);

  const handleImageUpload = useCallback((event) => {
    const file = event.target.files[0];
    if (!file) return;

    const allowedTypes = ['image/jpeg', 'image/png', 'image/gif', 'image/webp'];
    if (!allowedTypes.includes(file.type)) {
      setUploadError('Please upload an image in JPEG, PNG, GIF, or WebP format.');
      return;
    }

    const maxSizeInBytes = 10 * 1024 * 1024; // 10 MB
    if (file.size > maxSizeInBytes) {
      setUploadError('File size exceeds 10 MB.');
      return;
    }

    setUploadedImageFile(file);
    setUploadError(null);

    const reader = new FileReader();
    reader.onload = (e) => {
      setUploadedImage(e.target.result); // Set preview
    };
    reader.readAsDataURL(file);
  }, []);

  // Handles the RESULT of the analysis (called after backend responds)
  const handleUploadedImageAnalysis = useCallback((imageDataUrl, analysisResult) => {
    console.log("App: Handling uploaded image analysis result:", analysisResult);
    
    // Check essential data from the new endpoint
    if (!analysisResult || !analysisResult.processed_id || !analysisResult.embedding || !analysisResult.analysis) {
        console.error("App: Invalid analysis result received from backend for uploaded image (missing fields).", analysisResult);
        setUploadError("Received invalid analysis data from server.");
        setSnackbarMessage('Error: Received incomplete analysis data.');
        setSnackbarOpen(true);
        return;
    }
    
    // Construct metadata object for the uploaded artwork
    const uploadedMetadata = {
      id: analysisResult.processed_id,
      objectID: analysisResult.processed_id,
      title: analysisResult.original_filename || `Uploaded Image ${analysisResult.processed_id}`,
      artistDisplayName: "User Upload",
      department: "User Uploads",
      type: "uploaded",
      primaryImageSmall: imageDataUrl, // The preview URL from the frontend
      primaryImage: imageDataUrl,
      cachedImageUrl: imageDataUrl,
      // originalUrl is not directly needed here anymore if analysis is included
      objectDate: 'N/A', medium: 'N/A', classification: 'N/A',
      culture: 'N/A', objectURL: null,
      // analysis_files might not be returned by the new endpoint, TBD
    };

    const uploadedEmbedding = analysisResult.embedding;
    // Get the full analysis object directly from the response
    const newAnalysisData = analysisResult.analysis;
    
    console.log("App.jsx: Using combined analysis from upload endpoint:", newAnalysisData);

    // Set selectedArtworkData with metadata, embedding AND analysis
    setSelectedArtworkData({
        metadata: uploadedMetadata,
        embedding: uploadedEmbedding,
        analysis: newAnalysisData // Use the analysis data directly from the response
    });

    // Add to main artworks list
    setArtworksData(prev => {
      // Prevent adding duplicates if somehow the ID already exists
      if (prev.items.some(item => item.id === uploadedMetadata.id)) {
        console.warn(`App: Attempted to add duplicate uploaded artwork ID: ${uploadedMetadata.id}`);
        return prev; 
      }
      // Add to the beginning of the list
      const newItems = [uploadedMetadata, ...prev.items];
      const newEmbeddings = [uploadedEmbedding, ...prev.embeddings];
      return { items: newItems, embeddings: newEmbeddings };
    });
    console.log(`App: Added uploaded artwork ${uploadedMetadata.id} to artworksData.`);

    // Set GradCAM/Attention Map URLs using the analysis data (if available)
    // Note: The new endpoint might not return these directly in analysis.gradcam/analysis.attention yet
    // We might need to adjust the backend or handle this differently if maps are needed immediately.
    setGradcamMapUrl(newAnalysisData?.gradcam?.map_url || `${BACKEND_URL}/api/gradcam_image/${uploadedMetadata.id}?t=${Date.now()}`);
    setAttentionMapUrl(newAnalysisData?.attention?.map_url || `${BACKEND_URL}/api/attention_image/${uploadedMetadata.id}?t=${Date.now()}`);

    // Close dialog and clear state
    setShowUploadDialog(false);
    handleClearUploadedImage();

  }, [handleClearUploadedImage, BACKEND_URL]); // Removed uploadedImageFile dependency as it's used to get title only now

  // --- MOVED UP & DEFINED: Function to START the analysis --- 
  const handleAnalyzeUploadedImage = useCallback(async () => {
    if (!uploadedImageFile) {
      setUploadError('Please upload an image first.');
      return;
    }

    setUploadLoading(true);
    setUploadError(null);
    console.log('App.jsx: Sending image for analysis...');

    try {
      const formData = new FormData();
      formData.append('image', uploadedImageFile);

      console.log(`App.jsx: Uploading image: ${uploadedImageFile.name}, size: ${uploadedImageFile.size} bytes`);

      // *** Use the NEW backend endpoint ***
      const response = await fetch(`${BACKEND_URL}/api/upload_and_analyze`, { 
        method: 'POST',
        body: formData,
      });

      console.log(`App.jsx: Upload analysis response status: ${response.status}`);

      if (!response.ok) {
        const contentType = response.headers.get("content-type");
        let errorMsg = `Server Error: ${response.status}`;
        try {
            if (contentType && contentType.includes("application/json")) {
                const errorData = await response.json();
                errorMsg = errorData.error || errorMsg;
            } else {
                errorMsg = `${errorMsg} - ${await response.text()}`;
            }
        } catch (parseError) {
            console.error("Failed to parse error response", parseError);
        }
        throw new Error(errorMsg);
      }

      const analysisResult = await response.json();
      console.log('App.jsx: Received analysis results from new endpoint:', analysisResult);

      // Call the handler to process the results and update state
      // Pass the preview URL (uploadedImage) and the analysis result
      handleUploadedImageAnalysis(uploadedImage, analysisResult); 

      // Show success message
      setSnackbarMessage('Image analyzed successfully!');
      setSnackbarOpen(true);

    } catch (err) {
      console.error('App.jsx: Error analyzing uploaded image:', err);
      setUploadError(`Analysis failed: ${err.message}`);
       setSnackbarMessage(`Error: ${err.message}`); // Show error in snackbar too
       setSnackbarOpen(true);
    } finally {
      setUploadLoading(false);
    }
  // Dependencies for the function that STARTS analysis
  }, [uploadedImageFile, uploadedImage, BACKEND_URL, handleUploadedImageAnalysis, setSnackbarMessage, setSnackbarOpen, setUploadError, setUploadLoading]);

  // Drag and Drop Handlers
  const handleDragOver = useCallback((event) => {
      event.preventDefault();
      event.stopPropagation();
      setDragActive(true);
  }, []);

  const handleDragEnter = useCallback((event) => {
      event.preventDefault();
      event.stopPropagation();
      setDragActive(true);
  }, []);

  const handleDragLeave = useCallback((event) => {
      event.preventDefault();
      event.stopPropagation();
      setDragActive(false);
  }, []);

  const handleDrop = useCallback((event) => {
      event.preventDefault();
      event.stopPropagation();
      setDragActive(false);

      if (event.dataTransfer.files && event.dataTransfer.files.length > 0) {
          // Simulate file input event for validation and preview
          handleImageUpload({ target: { files: event.dataTransfer.files } });
          event.dataTransfer.clearData();
      }
  }, [handleImageUpload]);

  // Snackbar close handler
  const handleSnackbarClose = useCallback(() => {
      setSnackbarOpen(false);
  }, []);

  // --- END Upload Dialog Handlers ---

  // --- Derived State & Other Handlers ---
  const selectedArtworkId = selectedArtworkData ? selectedArtworkData.metadata?.id : null;
  const hasMoreArtworks = totalArtworks > 0 && artworksData.items.length < totalArtworks;

  // Debugging logs
  useEffect(() => {
    console.log("App: selectedArtworkData updated:", selectedArtworkData);
    console.log("App: artworksData items count:", artworksData.items.length);
  }, [selectedArtworkData, artworksData]);

  const handleShowUploadDialog = useCallback(() => {
    console.log("App: Opening upload dialog");
    setShowUploadDialog(true);
  }, []);

  const handleCloseUploadDialog = useCallback(() => {
    console.log("App: Closing upload dialog");
    setShowUploadDialog(false);
    handleClearUploadedImage(); // Also clear upload state on cancel
  }, [handleClearUploadedImage]);

  // IndexedDB Cleanup function (keep if needed, though unused currently)
  // const cleanupOldImages = async () => { ... };

  // --- MOVED UP: Fetch Artworks Page --- 
  const fetchArtworksPage = useCallback(async (page) => {
    console.log(`Fetching artworks page ${page} using /api/museum_artworks...`);
    setErrorLoading(null);
    try {
      const response = await fetch(`${BACKEND_URL}/api/museum_artworks?page=${page}&limit=${ARTWORKS_PER_PAGE}`);
      if (!response.ok) {
          const errorData = await response.json().catch(() => ({ error: "Failed to fetch page data" }));
          throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      console.log(`Received data for page ${page}:`, data);
      console.log(`API response for page ${page}: Metadata length = ${data.metadata?.length ?? 0}, Embeddings length = ${data.embeddings?.length ?? 0}`);

      // --- Process and cache images --- 
      const processedMetadata = [];
      const processedEmbeddings = [];
      const minLength = Math.min(data.metadata?.length || 0, data.embeddings?.length || 0);
      const imageFetchPromises = []; // Array to hold image fetch promises

      for (let i = 0; i < minLength; i++) {
          const artwork = data.metadata[i];
          const embedding = data.embeddings[i];
          const artworkId = artwork?.id || artwork?.objectID;

          if (!artwork || !artworkId || !embedding) {
              console.warn(`Skipping item at index ${i} on page ${page} due to missing data.`);
              continue;
          }
          artwork.id = artworkId; // Ensure consistent ID

          // Asynchronously fetch and cache image, but don't wait here
          const imageUrl = artwork.primaryImageSmall || artwork.primaryImage;
          if (imageUrl) {
             // Push promise to the array, store original index and artwork ID
             imageFetchPromises.push(
                 fetchAndCacheImage(imageUrl, artworkId).then(cachedUrl => ({ index: i, artworkId, cachedUrl }))
             );
          } else {
              artwork.cachedImageUrl = null; // Ensure property exists
          }
          processedMetadata.push(artwork); // Add metadata immediately (will update cachedUrl later)
          processedEmbeddings.push(embedding);
      }

      // --- Wait for all image fetches for this page to complete ---
      console.log(`[fetchArtworksPage ${page}] Waiting for ${imageFetchPromises.length} image fetch/cache promises...`);
      const imageResults = await Promise.allSettled(imageFetchPromises);
      console.log(`[fetchArtworksPage ${page}] Image fetch/cache promises settled, successful:`, 
                 imageResults.filter(r => r.status === 'fulfilled').length, 
                 "failed:", imageResults.filter(r => r.status === 'rejected').length);

      // --- Update processedMetadata with cached URLs ---
      imageResults.forEach(result => {
          if (result.status === 'fulfilled') {
              const { index, artworkId, cachedUrl } = result.value || {};
              // Find the item in processedMetadata using the original index or ID
              const itemIndexInProcessed = processedMetadata.findIndex(item => (item.id || item.objectID) === artworkId);
              if (itemIndexInProcessed !== -1) {
                 processedMetadata[itemIndexInProcessed].cachedImageUrl = cachedUrl;
                 console.log(`[fetchArtworksPage ${page}] Assigned cachedUrl to ID ${artworkId}: ${cachedUrl ? cachedUrl.substring(0,60)+'...' : 'null'}`);
              } else {
                 console.warn(`[fetchArtworksPage ${page}] Could not find processed item for ID ${artworkId} to assign cached URL.`);
              }
          } else if (result.status === 'rejected') {
               console.error(`[fetchArtworksPage ${page}] An image fetch promise failed:`, result.reason);
          }
      });
      // --- End update ---

      // Return processed data
      return {
          ...data, // Include total, page info from original response
          metadata: processedMetadata, // Now includes cachedImageUrls
          embeddings: processedEmbeddings
      };

    } catch (error) {
      console.error(`Error fetching page ${page}:`, error);
      setErrorLoading(error.message || 'Failed to load artwork data.');
      return null; // Return null on error
    }
  }, [BACKEND_URL, ARTWORKS_PER_PAGE, fetchAndCacheImage]); // Added fetchAndCacheImage

  // --- MOVED UP: Load More Artworks --- 
  const loadMoreArtworks = useCallback(async () => {
    console.log("%%% loadMoreArtworks START %%%");
    if (isLoadingMore || isLoadingInitial) return;
    
    const currentCount = artworksData.items.length;
    if (totalArtworks > 0 && currentCount >= totalArtworks) {
        console.log("No more artworks to load.");
        return;
    }

    setIsLoadingMore(true);
    const nextPage = currentPage + 1;
    const data = await fetchArtworksPage(nextPage); // Use fetchArtworksPage defined above

    if (data && data.metadata && data.embeddings && data.metadata.length === data.embeddings.length) {
        console.log(`[loadMoreArtworks] Embeddings received for page ${nextPage} (first 5):`, data.embeddings.slice(0, 5).map(e => e?.slice(0, 10)));
        setArtworksData(prevData => ({
            items: [...prevData.items, ...data.metadata],
            embeddings: [...prevData.embeddings, ...data.embeddings]
        }));
        setCurrentPage(data.page);
        if (data.total && data.total > totalArtworks) {
            setTotalArtworks(data.total);
        }
    } else if (data) {
         console.warn(`LoadMoreArtworks: Received data for page ${nextPage} but lengths mismatch or missing data. Not updating state.`);
    }
    setIsLoadingMore(false);
  }, [currentPage, totalArtworks, isLoadingMore, isLoadingInitial, artworksData, fetchArtworksPage]); // Added fetchArtworksPage dependency

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <BrowserRouter>
        <Routes>
          {/* Route for HomePage - outside the main layout */}
          <Route 
            path="/" 
            element={<HomePage artworksData={artworksData} />} 
          />

          {/* Route for the main application layout */}
          <Route 
            path="/*" 
            element={
              <Box sx={{ 
                display: 'flex', 
                minHeight: '100vh', 
                flexDirection: 'column',
                bgcolor: 'background.default'
              }}>
                <Box sx={{ 
                  display: 'flex', 
                  flex: 1,
                  position: 'relative',
                  overflow: 'hidden'
                }}>
                  <Sidebar 
                    artworks={artworksData.items} 
                    onArtworkSelect={handleArtworkSelect}
                    isLoadingMore={isLoadingMore}
                    hasMoreArtworks={hasMoreArtworks}
                    onLoadMore={loadMoreArtworks}
                    selectedArtworkId={selectedArtworkId}
                  />
                  <Box component="main" sx={{ flex: 1, p: 2, overflow: 'auto', height: '100vh' }}>
                     {/* REMOVED initial loading block */}
                     {errorLoading && ( // Show error FIRST if initial fetch failed
                        <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '80vh', flexDirection: 'column' }}>
                            <Alert severity="error" sx={{mb: 2}}>Error loading artwork data: {errorLoading}</Alert>
                             <Button variant="outlined" onClick={() => window.location.reload()}>Reload Page</Button>
                        </Box>
                     )}
                     {!errorLoading && ( // Only render routes if NO error
                    <Routes>
                      {/* Overview Page Route */}
                      <Route path="overview" element={<OverviewPage />} />

                      {/* Embeddings Page Route */}
                      <Route path="embeddings" element={
                         <EmbeddingProjection 
                          embeddings={artworksData.embeddings}
                          artworkMetadata={artworksData.items}
                          onArtworkSelect={handleArtworkSelect}
                             isLoading={isLoadingInitial || isLoadingMore}
                             selectedArtworkId={selectedArtworkId}
                        />
                      } />
                      
                      {/* Similarity Graph Route */}
                      <Route path="similarity" element={
                        <ArtSimilarityGraph 
                                embeddings={artworksData.embeddings}
                                artworkMetadata={artworksData.items}
                                onArtworkSelect={handleArtworkSelect}
                                selectedArtworkId={selectedArtworkId}
                                     isLoading={isLoadingInitial || isLoadingMore}
                        />
                      } />
                      
                      {/* Heatmap Route - Now uses SimilarityGallery */}
                      <Route path="heatmap" element={
                              <SimilarityGallery 
                                  embeddings={artworksData.embeddings} 
                                  artworkMetadata={artworksData.items} 
                                  selectedArtworkId={selectedArtworkId}
                                  onArtworkSelect={handleArtworkSelect} // Pass selection handler
                              />
                      } />
                      
                      {/* Attention Route */}
                      <Route path="attention" element={
                              <AttentionVisualizer 
                             selectedArtworkData={selectedArtworkData}
                        />
                      } />
                      
                      {/* Analysis Route */}
                      <Route path="analysis" element={
                        <StyleAnalysis 
                                artworks={artworksData.items}
                                embeddings={artworksData.embeddings}
                                selectedArtworkData={selectedArtworkData}
                                onArtworkSelect={handleArtworkSelect}
                                     isProcessingAnalysis={isProcessing}
                                     processingError={errorProcessing}
                                     showUploadDialog={showUploadDialog}
                                     onOpenUploadDialog={handleShowUploadDialog}
                                     onCloseUploadDialog={handleCloseUploadDialog}
                                     onUploadedImageAnalysis={handleUploadedImageAnalysis}
                        />
                      } />
                      
                         {/* Optional Redirect/Not Found */}
                         <Route index element={<Navigate to="overview" replace />} />
                    </Routes>
                     )}
                  </Box>
                  
                  <Box sx={{ width: 300, borderLeft: '1px solid #e0e0e0' }}>
                    <GlobalImagePanel 
                      selectedArtwork={selectedArtworkData ? {
                          ...selectedArtworkData.metadata,
                          imageUrl: selectedArtworkData.metadata.cachedImageUrl || selectedArtworkData.metadata.primaryImageSmall || selectedArtworkData.metadata.primaryImage
                      } : null}
                      baseUrl={BACKEND_URL}
                      onUploadClick={handleShowUploadDialog}
                      isLoading={isProcessing}
                    />
                  </Box>
                </Box>
              </Box>
            }
          />
        </Routes>
      </BrowserRouter>

      {/* Upload Dialog */}
      <Dialog
        open={showUploadDialog}
        onClose={handleCloseUploadDialog}
        aria-labelledby="upload-dialog-title"
      >
        <DialogTitle id="upload-dialog-title">{"Upload and Analyze Image"}</DialogTitle>
        <DialogContent>
          <Box
            sx={{ display: 'flex', flexDirection: 'column', gap: 2, minWidth: 300, p: 1 }}
            onDragOver={handleDragOver}
            onDragEnter={handleDragEnter}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            <input
              type="file" id="file-upload" accept="image/*"
              onChange={handleImageUpload} style={{ display: 'none' }} ref={fileInputRef}
            />
            <Button
                variant={dragActive ? "contained" : "outlined"}
              component="label"
              htmlFor="file-upload"
              startIcon={<FileUploadIcon />}
                sx={{
                    mt: 1,
                    borderStyle: dragActive ? 'dashed' : 'solid',
                    backgroundColor: dragActive ? 'action.hover' : 'transparent'
                 }}
             >
                {dragActive ? "Drop image here" : (uploadedImageFile ? `Selected: ${uploadedImageFile.name}` : "Choose or Drop Image")}
            </Button>
            
            {uploadedImage && (
              <Box sx={{ mt: 1, textAlign: 'center', border: '1px solid #eee', p: 1 }}>
                <Typography variant="caption">Preview:</Typography>
                <img 
                  src={uploadedImage} alt="Preview"
                  style={{ maxWidth: '100%', maxHeight: '200px', objectFit: 'contain', display: 'block', margin: 'auto' }}
                />
                 <Button size="small" onClick={handleClearUploadedImage} sx={{mt: 1}}>Clear</Button>
              </Box>
            )}
            {uploadError && (
              <Alert severity="error" sx={{ mt: 1 }}>{uploadError}</Alert>
            )}
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseUploadDialog}>Cancel</Button>
          <Button
            onClick={handleAnalyzeUploadedImage}
            color="primary" variant="contained"
            disabled={!uploadedImageFile || uploadLoading}
            startIcon={uploadLoading ? <CircularProgress size={20} /> : null}
          >
            {uploadLoading ? 'Analyzing...' : 'Analyze'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Snackbar */}
      <Snackbar
        open={snackbarOpen}
        autoHideDuration={6000}
        onClose={handleSnackbarClose}
        message={snackbarMessage}
        action={
          <IconButton size="small" color="inherit" onClick={handleSnackbarClose}>
            <CloseIcon fontSize="small" />
          </IconButton>
        }
      />
    </ThemeProvider>
  );
};

export default App;
