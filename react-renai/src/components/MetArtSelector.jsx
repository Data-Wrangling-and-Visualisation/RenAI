import React, { useEffect, useState } from 'react';
import { Grid, Paper, Typography, CircularProgress } from '@mui/material';

const MetArtSelector = ({ onSelect }) => {
  const [objects, setObjects] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch('/api/objects')
      .then(res => res.json())
      .then(data => {
        setObjects(data.objectIDs.slice(0, 100)); // First 100 objects
        setLoading(false);
      });
  }, []);

  return (
    <Paper sx={{ p: 3, mt: 2 }}>
      <Typography variant="h6" gutterBottom>
        Select Artwork from Metropolitan Museum
      </Typography>
      {loading ? (
        <CircularProgress />
      ) : (
        <Grid container spacing={2}>
          {objects.map(objectID => (
            <Grid item xs={4} sm={3} md={2} key={objectID}>
              <img 
                src={`https://collectionapi.metmuseum.org/public/collection/v1/objects/${objectID}/primary-image`}
                alt="Artwork"
                style={{ 
                  width: '100%',
                  cursor: 'pointer',
                  borderRadius: '4px'
                }}
                onClick={() => onSelect(objectID)}
              />
            </Grid>
          ))}
        </Grid>
      )}
    </Paper>
  );
};
