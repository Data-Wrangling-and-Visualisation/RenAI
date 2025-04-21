import React, { useEffect, useState, useMemo, useCallback } from 'react';
import { 
  Box, Paper, Typography, Grid, FormControl,
  InputLabel, Select, MenuItem, CircularProgress,
  Tooltip, Chip, Alert, Card, CardActionArea, CardMedia, CardContent
} from '@mui/material';
import { debounce } from 'lodash'; // Use debounce for resizing or intensive calcs if needed

// Function to calculate cosine similarity between two vectors
const cosineSimilarity = (vecA, vecB) => {
  if (!vecA || !vecB || vecA.length !== vecB.length) return 0;
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < vecA.length; i++) {
    dotProduct += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }
  normA = Math.sqrt(normA);
  normB = Math.sqrt(normB);
  if (normA === 0 || normB === 0) return 0;
  // Clamp similarity slightly below 1 to avoid potential floating point issues if needed
  // return Math.min(0.99999, dotProduct / (normA * normB)); 
  return dotProduct / (normA * normB);
};

// Helper to get a stable sort key from metadata
const getSortKey = (meta, sortBy) => {
    switch(sortBy) {
        case 'classification': return meta.classification || 'zzzzz'; // Push unknowns to end
        case 'date': 
            const yearMatch = String(meta.objectDate).match(/\b(\d{4})\b/);
            return yearMatch ? parseInt(yearMatch[1], 10) : 9999; // Push unknowns/unparsable to end
        case 'title': return meta.title || 'zzzzz';
        case 'id':
        default: return meta.id || meta.objectID || 'zzzzz';
    }
};

// Number of neighbors to show in the heatmap
const K_NEIGHBORS = 50; 

// Renaming component to better reflect its purpose
const SimilarityGallery = ({ embeddings = [], artworkMetadata = [], selectedArtworkId, onArtworkSelect }) => {
  const [loading, setLoading] = useState(false);
  const [neighbors, setNeighbors] = useState([]);
  const [calculationError, setCalculationError] = useState(null);
  
  useEffect(() => {
    setNeighbors([]); // Clear previous neighbors
    setCalculationError(null);
    
    if (!selectedArtworkId) {
      setCalculationError("Select an artwork to view similar items.");
      setLoading(false);
      return;
    }

    setLoading(true);

    if (!embeddings || !artworkMetadata || embeddings.length === 0 || artworkMetadata.length === 0 || embeddings.length !== artworkMetadata.length) {
      console.warn("SimilarityGallery: Invalid or mismatched embeddings/metadata.");
      setCalculationError("Invalid artwork data for calculation.");
      setLoading(false);
      return;
    }
    
    const selectedIndex = artworkMetadata.findIndex(meta => (meta.id || meta.objectID) === selectedArtworkId);
    if (selectedIndex === -1) {
        console.warn(`SimilarityGallery: Selected artwork ID ${selectedArtworkId} not found.`);
        setCalculationError(`Selected artwork ID ${selectedArtworkId} not found.`);
        setLoading(false);
        return;
    }
    const selectedEmbedding = embeddings[selectedIndex];
    if (!selectedEmbedding) {
        console.warn(`SimilarityGallery: Embedding not found for ${selectedArtworkId}.`);
        setCalculationError(`Embedding not found for selected artwork ID ${selectedArtworkId}.`);
        setLoading(false);
        return;
    }

    // Use setTimeout to allow UI to update before calculation
    const timerId = setTimeout(() => {
      try {
        console.log(`SimilarityGallery: Calculating Top ${K_NEIGHBORS} neighbors for ${selectedArtworkId}...`);
        console.log(`SimilarityGallery: Selected Embedding (first 10 dims):`, selectedEmbedding?.slice(0, 10));
        let sampleCompared = false;

        const similarities = embeddings.map((emb, idx) => {
            const meta = artworkMetadata[idx];
            const id = meta?.id || meta?.objectID;
            if (!emb || !id || idx === selectedIndex) return null; 
            
            const similarityScore = cosineSimilarity(selectedEmbedding, emb);

            if (!sampleCompared && idx !== selectedIndex) {
                 console.log(`SimilarityGallery: Comparing with ID ${id} Embedding (first 10 dims):`, emb?.slice(0, 10));
                 console.log(`SimilarityGallery: Calculated Similarity Score for ${id}:`, similarityScore);
                 sampleCompared = true;
            }

            return {
                metadata: meta, // Keep the full metadata
                similarity: similarityScore
            };
        }).filter(item => item !== null); 

        similarities.sort((a, b) => b.similarity - a.similarity);
        const topKNeighbors = similarities.slice(0, K_NEIGHBORS);

        console.log(`SimilarityGallery: Top ${Math.min(K_NEIGHBORS, 5)} Neighbors (before setting state):`, topKNeighbors.slice(0, 5));

        console.log(`SimilarityGallery: Found ${topKNeighbors.length} neighbors for ${selectedArtworkId}.`);
        setNeighbors(topKNeighbors);
        setCalculationError(null); 

      } catch (error) {
          console.error("SimilarityGallery: Error during calculation:", error);
          setCalculationError(`Calculation failed: ${error.message}`);
          setNeighbors([]);
      } finally {
        setLoading(false);
      }
    }, 50); 

    return () => clearTimeout(timerId);

  }, [embeddings, artworkMetadata, selectedArtworkId]); 
  
  const selectedArtworkTitle = useMemo(() => {
      if (!selectedArtworkId) return null;
      const meta = artworkMetadata.find(m => (m.id || m.objectID) === selectedArtworkId);
      return meta?.title || selectedArtworkId;
  }, [selectedArtworkId, artworkMetadata]);

  // Handle click on a neighbor card
  const handleNeighborClick = useCallback((neighborId) => {
      if (onArtworkSelect) {
          onArtworkSelect(neighborId);
      }
  }, [onArtworkSelect]);

  return (
    <Paper sx={{ p: 2, height: 'calc(100vh - 64px - 32px)', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      <Typography variant="h5" gutterBottom>
        Similar Artworks Gallery
      </Typography>
      {selectedArtworkId ? (
           <Typography variant="subtitle1" sx={{ mb: 2 }}>
               Showing Top {K_NEIGHBORS} artworks similar to: <strong>{selectedArtworkTitle}</strong> (ID: {selectedArtworkId})
            </Typography>
      ) : (
           <Typography variant="subtitle1" sx={{ mb: 2 }}>
               Select an artwork to see similar items.
            </Typography>
      )}
      
      <Box sx={{ flexGrow: 1, overflowY: 'auto', position: 'relative' }}> {/* Make this Box scrollable */}
        {loading && (
          <Box sx={{ position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', bgcolor: 'rgba(255,255,255,0.7)', zIndex: 10 }}>
            <CircularProgress />
            <Typography sx={{mt: 2}}>Calculating similarities...</Typography>
          </Box>
        )}
        {!loading && calculationError && (
             <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%', p: 2 }}>
                <Alert severity="warning" sx={{width: '100%'}}>{calculationError}</Alert>
             </Box>
        )}
        {!loading && !calculationError && neighbors.length > 0 && (
           <Grid container spacing={2} sx={{ p: 1 }}> {/* Add padding to grid container */}
               {neighbors.map((neighbor) => {
                   const neighborId = neighbor.metadata.id || neighbor.metadata.objectID;
                   const imageUrl = neighbor.metadata.cachedImageUrl || neighbor.metadata.primaryImageSmall;
                   return (
                       <Grid item xs={6} sm={4} md={3} lg={2} key={neighborId}> {/* Adjust grid sizing */}
                           <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
                               <CardActionArea 
                                   onClick={() => handleNeighborClick(neighborId)} 
                                   sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column' }}
                                >
                                   <CardMedia
                                       component="img"
                                       sx={{ 
                                           height: 140, // Fixed height for image area
                                           objectFit: 'contain', // Ensure image fits without stretching
                                           bgcolor: '#f0f0f0' // Background for empty space
                                       }}
                                       image={imageUrl || './placeholder.png'} // Use a placeholder if no image
                                       alt={neighbor.metadata.title || 'Artwork'}
                                       onError={(e) => { e.target.onerror = null; e.target.src='./placeholder.png'; }} // Fallback on error
                                   />
                                   <CardContent sx={{ flexGrow: 1, p: 1 }}>
                                       <Tooltip title={neighbor.metadata.title || 'Unknown Title'}>
                                            <Typography gutterBottom variant="caption" component="div" noWrap>
                                                {neighbor.metadata.title || 'Unknown Title'}
                                            </Typography>
                                       </Tooltip>
                                       <Typography variant="body2" color="text.secondary">
                                           Sim: {neighbor.similarity.toFixed(3)}
                                       </Typography>
                                   </CardContent>
                               </CardActionArea>
                           </Card>
                       </Grid>
                   );
               })}
           </Grid>
        )}
         {!loading && !calculationError && neighbors.length === 0 && selectedArtworkId && (
             <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%', p: 2 }}>
                 <Typography variant="body1" sx={{ textAlign: 'center', mt: 4 }}>
                     No similarity data calculated or no neighbors found.
                 </Typography>
              </Box>
        )}
      </Box>
    </Paper>
  );
};

export default SimilarityGallery;
